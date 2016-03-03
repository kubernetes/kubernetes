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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## kubectl daemonset-rolling-update

Perform a rolling update of the given DaemonSet.

### Synopsis


Perform a rolling update of the given DaemonSet.

Replaces the specified daemon set with a new daemon set by creating a daemon set with the same selector and delering the
existing pods. The new-daemonset.json must specify the same namespace as the
existing daemon set

```
kubectl daemonset-rolling-update OLD_DS_NAME -f NEW_DS_SPEC
```

### Examples

```
# Update the daemon set ds-v1 into ds-v1 and recreate associated pod.
$ kubectl daemonset-rolling-update ds-v1 -f ds-v2.json

# Update daemon set ds-v1 using JSON data passed into stdin.
$ cat ds-v2.json | kubectl daemonset-rolling-update ds-v1 -f -

# Abort and reverse an existing rollout in progress (from ds-v1 to ds-v2).
$ kubectl daemonset-rolling-update ds-v1 ds-v2 --rollback

```

### Options

```
      --delete-interval=10s: Time delay between daemon set creation and deletion of old one. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
      --dry-run[=false]: If true, print out the changes that would be made, but don't actually make them.
  -f, --filename=[]: Filename or URL to file to use to create the new daemon set.
      --no-headers[=false]: When using the default output, don't print headers.
  -o, --output="": Output format. One of: json|yaml|wide|name|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://releases.k8s.io/HEAD/docs/user-guide/jsonpath.md].
      --output-version="": Output the formatted object with the given group version (for ex: 'extensions/v1beta1').
      --recreate-interval=0: Time to wait between each pod recreation. You may not need that because the update wait for pods to be ready before deleting the next one. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
      --rollback[=false]: If true, this is a request to abort an existing rollout that is partially rolled out. It effectively reverses current and next and runs a rollout
      --schema-cache-dir="~/.kube/schema": If non-empty, load/store cached API schemas in this directory, default is '$HOME/.kube/schema'
  -a, --show-all[=false]: When printing, show all resources (default hide terminated pods.)
      --show-labels[=false]: When printing, show all labels as the last column (default hide labels column)
      --sort-by="": If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. '{.metadata.name}'). The field in the API resource specified by this JSONPath expression must be an integer or a string.
      --template="": Template string or path to template file to use when -o=go-template, -o=go-template-file. The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview].
      --timeout=5m0s: Max time to wait for the full update before giving up. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
      --validate[=true]: If true, use a schema to validate the input before sending it
```

### Options inherited from parent commands

```
      --alsologtostderr[=false]: log to standard error as well as files
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

###### Auto generated by spf13/cobra on 14-Mar-2016

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubectl/kubectl_daemonset-rolling-update.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
