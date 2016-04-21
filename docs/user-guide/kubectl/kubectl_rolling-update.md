<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.2/docs/user-guide/kubectl/kubectl_rolling-update.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## kubectl rolling-update

Perform a rolling update of the given ReplicationController.

### Synopsis


Perform a rolling update of the given ReplicationController.

Replaces the specified replication controller with a new replication controller by updating one pod at a time to use the
new PodTemplate. The new-controller.json must specify the same namespace as the
existing replication controller and overwrite at least one (common) label in its replicaSelector.

```
kubectl rolling-update OLD_CONTROLLER_NAME ([NEW_CONTROLLER_NAME] --image=NEW_CONTAINER_IMAGE | -f NEW_CONTROLLER_SPEC)
```

### Examples

```
# Update pods of frontend-v1 using new replication controller data in frontend-v2.json.
kubectl rolling-update frontend-v1 -f frontend-v2.json

# Update pods of frontend-v1 using JSON data passed into stdin.
cat frontend-v2.json | kubectl rolling-update frontend-v1 -f -

# Update the pods of frontend-v1 to frontend-v2 by just changing the image, and switching the
# name of the replication controller.
kubectl rolling-update frontend-v1 frontend-v2 --image=image:v2

# Update the pods of frontend by just changing the image, and keeping the old name.
kubectl rolling-update frontend --image=image:v2

# Abort and reverse an existing rollout in progress (from frontend-v1 to frontend-v2).
kubectl rolling-update frontend-v1 frontend-v2 --rollback

```

### Options

```
      --container="": Container name which will have its image upgraded. Only relevant when --image is specified, ignored otherwise. Required when using --image on a multi-container pod
      --deployment-label-key="deployment": The key to use to differentiate between two different controllers, default 'deployment'.  Only relevant when --image is specified, ignored otherwise
      --dry-run[=false]: If true, print out the changes that would be made, but don't actually make them.
  -f, --filename=[]: Filename or URL to file to use to create the new replication controller.
      --image="": Image to use for upgrading the replication controller. Must be distinct from the existing image (either new image or new image tag).  Can not be used with --filename/-f
      --image-pull-policy="": Explicit policy for when to pull container images. Required when --image is same as existing image, ignored otherwise.
      --include-extended-apis[=true]: If true, include definitions of new APIs via calls to the API server. [default true]
      --no-headers[=false]: When using the default output, don't print headers.
  -o, --output="": Output format. One of: json|yaml|wide|name|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://releases.k8s.io/HEAD/docs/user-guide/jsonpath.md].
      --output-version="": Output the formatted object with the given group version (for ex: 'extensions/v1beta1').
      --poll-interval=3s: Time delay between polling for replication controller status after the update. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
      --rollback[=false]: If true, this is a request to abort an existing rollout that is partially rolled out. It effectively reverses current and next and runs a rollout
      --schema-cache-dir="~/.kube/schema": If non-empty, load/store cached API schemas in this directory, default is '$HOME/.kube/schema'
  -a, --show-all[=false]: When printing, show all resources (default hide terminated pods.)
      --show-labels[=false]: When printing, show all labels as the last column (default hide labels column)
      --sort-by="": If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. '{.metadata.name}'). The field in the API resource specified by this JSONPath expression must be an integer or a string.
      --template="": Template string or path to template file to use when -o=go-template, -o=go-template-file. The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview].
      --timeout=5m0s: Max time to wait for a replication controller to update before giving up. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
      --update-period=1m0s: Time to wait between updating pods. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
      --validate[=true]: If true, use a schema to validate the input before sending it
```

### Options inherited from parent commands

```
      --alsologtostderr[=false]: log to standard error as well as files
      --as="": Username to impersonate for the operation.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client certificate file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --insecure-skip-tls-verify[=false]: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log-backtrace-at=:0: when logging hits line file:N, emit a stack trace
      --log-dir="": If non-empty, write log files in this directory
      --log-flush-frequency=5s: Maximum number of seconds between log flushes
      --logtostderr[=true]: log to standard error instead of files
      --match-server-version[=false]: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --password="": Password for basic authentication to the API server.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --username="": Username for basic authentication to the API server.
      --v=0: log level for V logs
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging
```

### SEE ALSO

* [kubectl](kubectl.md)	 - kubectl controls the Kubernetes cluster manager

###### Auto generated by spf13/cobra on 21-Apr-2016

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubectl/kubectl_rolling-update.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
