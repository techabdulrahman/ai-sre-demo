import argparse
import json
import os
import sys
from functools import wraps

from google import genai
from google.genai import types
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

try:
    from sre_tools import SRE_TOOLS
except ModuleNotFoundError as exc:
    if exc.name != "sre_tools":
        raise
    SRE_TOOLS = ()
    _SRE_TOOLS_IMPORT_ERROR = "sre_tools.py is missing; GitHub and Grafana tools are unavailable."
else:
    _SRE_TOOLS_IMPORT_ERROR = None


SYSTEM_PROMPT = """You are an AI SRE assistant investigating a failing Kubernetes pod.
Investigate in this exact order:
1. Inspect Kubernetes pod status, events, and logs.
2. Inspect Grafana alerts and metrics, including memory usage versus the limit
   and restart counts, to determine whether an OOM is a gradual leak or a spike.
3. Inspect the latest GitHub Actions workflow runs and failed-step details to
   determine whether a recent deployment caused the problem.
4. Only after those checks, state the evidence-based root cause and recommend a
   fix. Clearly state any data source that could not be checked and do not claim
   that it was checked.

All tools are read-only. Never restart, patch, roll back, or otherwise change
the cluster or deployment. Before any remediation, ask for explicit
confirmation in the terminal. In this diagnostic-only program, provide a
recommendation and wait for the operator rather than executing remediation.
"""


_FAILURES = set()
_CONFIG = {
    "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
    "GITHUB_REPO": os.environ.get("GITHUB_REPO", ""),
    "GRAFANA_URL": os.environ.get("GRAFANA_URL", ""),
    "GRAFANA_TOKEN": os.environ.get("GRAFANA_TOKEN", ""),
    "GRAFANA_DS_UID": os.environ.get("GRAFANA_DS_UID", ""),
}


def _report_tool_failure(name, error):
    lower_name = name.lower()
    if "grafana" in lower_name or "prometheus" in lower_name:
        source = "Grafana"
    elif "github" in lower_name or "workflow" in lower_name or "failed_step" in lower_name:
        source = "GitHub Actions"
    else:
        source = name
    _FAILURES.add(source)
    return f"{source} check failed for {name}: {error}"


def _redacted_arguments(arguments):
    return {
        key: "[REDACTED]" if any(secret in key.lower() for secret in ("token", "secret", "password")) else value
        for key, value in arguments.items()
    }


def _logged_sre_tool(tool):
    @wraps(tool)
    def call_tool(*args, **kwargs):
        try:
            print(
                f"Tool call: {tool.__name__} "
                f"{json.dumps({'args': args, 'kwargs': _redacted_arguments(kwargs)}, default=str)}",
                flush=True,
            )
            missing = []
            lower_name = tool.__name__.lower()
            if "grafana" in lower_name or "prometheus" in lower_name:
                required = ("GRAFANA_URL", "GRAFANA_TOKEN", "GRAFANA_DS_UID")
                missing = [name for name in required if not _CONFIG[name]]
            elif "github" in lower_name or "workflow" in lower_name or "failed_step" in lower_name:
                required = ("GITHUB_TOKEN", "GITHUB_REPO")
                missing = [name for name in required if not _CONFIG[name]]
            if missing:
                return _report_tool_failure(
                    tool.__name__,
                    "missing environment variable(s): " + ", ".join(missing),
                )
            return tool(*args, **kwargs)
        except Exception as exc:
            return _report_tool_failure(tool.__name__, str(exc))

    return call_tool


def _pod_or_default(pod):
    return pod or POD_NAME


def _resolve_pod_name(pod_name, namespace):
    try:
        _CORE_V1.read_namespaced_pod(pod_name, namespace)
        return pod_name
    except ApiException as exc:
        if exc.status != 404:
            raise
    matches = _CORE_V1.list_namespaced_pod(
        namespace,
        label_selector=f"app={pod_name}",
    ).items
    if not matches:
        raise ValueError(
            f"No pod named {pod_name!r} or pod with label app={pod_name!r} in namespace {namespace!r}."
        )
    return sorted(matches, key=lambda item: item.metadata.name)[0].metadata.name


def get_pod_status(pod: str = "", namespace: str = "") -> str:
    """Read the selected pod's phase, container states, restart counts, and resource limits."""
    requested_pod = _pod_or_default(pod)
    pod_namespace = namespace or NAMESPACE
    print(
        f"Tool call: get_pod_status "
        f"{json.dumps({'pod': requested_pod, 'namespace': pod_namespace})}",
        flush=True,
    )
    try:
        pod_name = _resolve_pod_name(requested_pod, pod_namespace)
        result = _CORE_V1.read_namespaced_pod(pod_name, pod_namespace)
        status = result.status
        containers = []
        for container_status in status.container_statuses or []:
            state = container_status.state
            waiting = state.waiting.reason if state and state.waiting else None
            terminated = state.terminated.reason if state and state.terminated else None
            containers.append(
                {
                    "name": container_status.name,
                    "ready": container_status.ready,
                    "restart_count": container_status.restart_count,
                    "waiting_reason": waiting,
                    "terminated_reason": terminated,
                    "last_terminated_reason": (
                        container_status.last_state.terminated.reason
                        if container_status.last_state
                        and container_status.last_state.terminated
                        else None
                    ),
                }
            )
        limits = {
            container.name: (container.resources.limits or {})
            for container in result.spec.containers
        }
        return json.dumps(
            {
                "name": result.metadata.name,
                "namespace": result.metadata.namespace,
                "phase": status.phase,
                "reason": status.reason,
                "message": status.message,
                "container_statuses": containers,
                "resource_limits": limits,
            },
            default=str,
        )
    except ApiException as exc:
        return f"Kubernetes could not read pod {pod_namespace}/{requested_pod}: {exc}"
    except ValueError as exc:
        return str(exc)


def get_pod_events(pod: str = "", namespace: str = "") -> str:
    """Read recent Kubernetes events associated with the selected pod."""
    requested_pod = _pod_or_default(pod)
    pod_namespace = namespace or NAMESPACE
    print(
        f"Tool call: get_pod_events "
        f"{json.dumps({'pod': requested_pod, 'namespace': pod_namespace})}",
        flush=True,
    )
    try:
        pod_name = _resolve_pod_name(requested_pod, pod_namespace)
        events = _CORE_V1.list_namespaced_event(
            pod_namespace,
            field_selector=f"involvedObject.name={pod_name}",
        )
        return json.dumps(
            [
                {
                    "type": event.type,
                    "reason": event.reason,
                    "message": event.message,
                    "count": event.count,
                    "last_timestamp": event.last_timestamp or event.event_time,
                }
                for event in sorted(
                    events.items,
                    key=lambda item: str(
                        item.last_timestamp or item.event_time or item.metadata.creation_timestamp or ""
                    ),
                    reverse=True,
                )[:30]
            ],
            default=str,
        )
    except ApiException as exc:
        return f"Kubernetes could not read events for {pod_namespace}/{requested_pod}: {exc}"
    except ValueError as exc:
        return str(exc)


def get_pod_logs(pod: str = "", namespace: str = "") -> str:
    """Read recent logs from each container in the selected pod."""
    requested_pod = _pod_or_default(pod)
    pod_namespace = namespace or NAMESPACE
    print(
        f"Tool call: get_pod_logs "
        f"{json.dumps({'pod': requested_pod, 'namespace': pod_namespace})}",
        flush=True,
    )
    try:
        pod_name = _resolve_pod_name(requested_pod, pod_namespace)
        result = _CORE_V1.read_namespaced_pod(pod_name, pod_namespace)
        logs = {}
        for container in result.spec.containers:
            try:
                logs[container.name] = _CORE_V1.read_namespaced_pod_log(
                    pod_name,
                    pod_namespace,
                    container=container.name,
                    tail_lines=100,
                    timestamps=True,
                )
            except ApiException as exc:
                logs[container.name] = f"Could not read logs: {exc}"
        return json.dumps(logs, default=str)
    except ApiException as exc:
        return f"Kubernetes could not read logs for {pod_namespace}/{requested_pod}: {exc}"
    except ValueError as exc:
        return str(exc)


def _rule_based_fallback():
    """Summarize Kubernetes evidence when Gemini cannot complete an investigation."""
    try:
        pod_name = _resolve_pod_name(POD_NAME, NAMESPACE)
        pod = _CORE_V1.read_namespaced_pod(pod_name, NAMESPACE)
    except ApiException as exc:
        return (
            f"Gemini analysis failed and Kubernetes fallback could not read "
            f"{NAMESPACE}/{POD_NAME}: {exc}"
        )
    except ValueError as exc:
        return f"Gemini analysis failed and Kubernetes fallback could not locate the pod: {exc}"

    conditions = []
    suggestions = []
    for container_status in pod.status.container_statuses or []:
        state = container_status.state
        last_terminated = container_status.last_state.terminated if container_status.last_state else None
        waiting_reason = state.waiting.reason if state and state.waiting else None
        reason = (
            "OOMKilled"
            if last_terminated and last_terminated.reason == "OOMKilled"
            else waiting_reason
            or (state.terminated.reason if state and state.terminated else None)
            or (last_terminated.reason if last_terminated else None)
        )
        if reason == "ImagePullBackOff" or reason == "ErrImagePull":
            conditions.append(f"{container_status.name}: {reason}")
            suggestions.append("verify the image name, tag, registry access, and image pull credentials")
        elif reason == "CrashLoopBackOff":
            conditions.append(f"{container_status.name}: CrashLoopBackOff")
            suggestions.append("inspect the container logs and startup configuration before considering a restart")
        elif reason == "OOMKilled":
            conditions.append(f"{container_status.name}: OOMKilled")
            suggestions.append("compare memory usage with the container limit and determine leak versus spike before changing limits")

    if not conditions and pod.status.phase != "Running":
        conditions.append(f"pod phase: {pod.status.phase}")
        suggestions.append("inspect the pod events and container logs to identify the underlying failure")
    if not conditions:
        conditions.append("no ImagePullBackOff, CrashLoopBackOff, or OOMKilled state detected")
        suggestions.append("review the pod events and application health checks")

    checked = "Grafana and GitHub Actions were not analyzed by the rule-based fallback."
    if _FAILURES:
        checked = "Unavailable checks: " + ", ".join(sorted(_FAILURES)) + "."
    return (
        f"Rule-based Kubernetes fallback for {NAMESPACE}/{POD_NAME}: "
        f"{'; '.join(conditions)}. Suggested next step: "
        f"{'; '.join(dict.fromkeys(suggestions))}. {checked} "
        "No remediation was performed; ask for terminal confirmation before any action."
    )


def main():
    parser = argparse.ArgumentParser(description="Diagnose a Kubernetes pod with AI SRE tools.")
    parser.add_argument("--pod", default="demo-app", help="Pod name to investigate (default: demo-app)")
    parser.add_argument("--namespace", default="demo", help="Kubernetes namespace (default: demo)")
    args = parser.parse_args()

    global POD_NAME, NAMESPACE, _CORE_V1
    POD_NAME = args.pod
    NAMESPACE = args.namespace

    try:
        config.load_kube_config()
    except config.ConfigException:
        config.load_incluster_config()
    _CORE_V1 = client.CoreV1Api()

    tools = [get_pod_status, get_pod_events, get_pod_logs]
    tools.extend(_logged_sre_tool(tool) for tool in SRE_TOOLS)
    if _SRE_TOOLS_IMPORT_ERROR:
        _FAILURES.update(("GitHub Actions", "Grafana"))
        print(_SRE_TOOLS_IMPORT_ERROR, file=sys.stderr)

    try:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.")
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        prompt = (
            f"Investigate pod {POD_NAME} in namespace {NAMESPACE}. "
            "Follow the required investigation order, use the available read-only "
            "tools, then report evidence, root cause, and a recommended fix."
        )
        response = genai.Client(api_key=api_key).models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
            ),
        )
        text = response.text
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        print(text)
        if _FAILURES:
            print("Could not check: " + ", ".join(sorted(_FAILURES)) + ".")
    except Exception as exc:
        print(f"Gemini investigation unavailable; using rule-based fallback: {exc}", file=sys.stderr)
        print(_rule_based_fallback())


if __name__ == "__main__":
    main()
