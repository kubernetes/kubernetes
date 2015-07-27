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

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/kubectl/kubectl_proxy.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## kubectl proxy

Run a proxy to the Kubernetes API server

### Synopsis


To proxy all of the kubernetes api and nothing else, use:

kubectl proxy --api-prefix=/

To proxy only part of the kubernetes api and also some static files:

kubectl proxy --www=/my/files --www-prefix=/static/ --api-prefix=/api/

The above lets you 'curl localhost:8001/api/v1/pods'.

To proxy the entire kubernetes api at a different root, use:

kubectl proxy --api-prefix=/custom/

The above lets you 'curl localhost:8001/custom/api/v1/pods'


```
kubectl proxy [--port=PORT] [--www=static-dir] [--www-prefix=prefix] [--api-prefix=prefix]
```

### Examples

```
// Run a proxy to kubernetes apiserver on port 8011, serving static content from ./local/www/
$ kubectl proxy --port=8011 --www=./local/www/

// Run a proxy to kubernetes apiserver on an arbitrary local port.
// The chosen port for the server will be output to stdout.
$ kubectl proxy --port=0

// Run a proxy to kubernetes apiserver, changing the api prefix to k8s-api
// This makes e.g. the pods api available at localhost:8011/k8s-api/v1/pods/
$ kubectl proxy --api-prefix=/k8s-api
```

### Options

```
      --accept-hosts="^localhost$,^127\\.0\\.0\\.1$,^\\[::1\\]$": Regular expression for hosts that the proxy should accept.
      --accept-paths="^/.*": Regular expression for paths that the proxy should accept.
      --api-prefix="/api/": Prefix to serve the proxied API under.
      --disable-filter=false: If true, disable request filtering in the proxy. This is dangerous, and can leave you vulnerable to XSRF attacks.  Use with caution.
  -h, --help=false: help for proxy
  -p, --port=8001: The port on which to run the proxy. Set to 0 to pick a random port.
      --reject-methods="POST,PUT,PATCH": Regular expression for HTTP methods that the proxy should reject.
      --reject-paths="^/api/.*/exec,^/api/.*/run": Regular expression for paths that the proxy should reject.
  -w, --www="": Also serve static files from the given directory under the specified prefix.
  -P, --www-prefix="/static/": Prefix to serve static files under, if static file directory is specified.
```

### Options inherited from parent commands

```
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log-backtrace-at=:0: when logging hits line file:N, emit a stack trace
      --log-dir=: If non-empty, write log files in this directory
      --log-flush-frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --password="": Password for basic authentication to the API server.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --username="": Username for basic authentication to the API server.
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging
```

### SEE ALSO

* [kubectl](kubectl.md)	 - kubectl controls the Kubernetes cluster manager

###### Auto generated by spf13/cobra at 2015-07-14 00:11:42.957150329 +0000 UTC


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubectl/kubectl_proxy.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
