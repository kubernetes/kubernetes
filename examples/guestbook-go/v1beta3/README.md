## GuestBook v1beta3 example

This example shows how to build a simple, multi-tier web application using Kubernetes and Docker.

The example consists of:
- A web frontend
- A redis master (for storage and a replicated set of redis slaves)

The web front end interacts with the redis master via javascript redis API calls.

The v1beta3 API is not enabled by default. The kube-apiserver process needs to run with the --runtime_config=api/v1beta3 argument. Use the following command to enable it:
$sudo sed -i 's|KUBE_API_ARGS="|KUBE_API_ARGS="--runtime_config=api/v1beta3 |' /etc/kubernetes/apiserver


