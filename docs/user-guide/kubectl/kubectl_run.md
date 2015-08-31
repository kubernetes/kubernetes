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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/kubectl/kubectl_run.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## kubectl run

Run a particular image on the cluster.

### Synopsis


Create and run a particular image, possibly replicated.
Creates a replication controller to manage the created container(s).

```
kubectl run NAME --image=image [--port=port] [--replicas=replicas] [--dry-run=bool] [--overrides=inline-json]
```

### Examples

```
# Starts a single instance of nginx.
$ kubectl run nginx --image=nginx

# Starts a replicated instance of nginx.
$ kubectl run nginx --image=nginx --replicas=5

# Dry run. Print the corresponding API objects without creating them.
$ kubectl run nginx --image=nginx --dry-run

# Start a single instance of nginx, but overload the spec of the replication controller with a partial set of values parsed from JSON.
$ kubectl run nginx --image=nginx --overrides='{ "apiVersion": "v1", "spec": { ... } }'

# Start a single instance of nginx and keep it in the foreground, don't restart it if it exits.
$ kubectl run -i -tty nginx --image=nginx --restart=Never

# Start the nginx container using the default command, but use custom arguments (arg1 .. argN) for that command.
$ kubectl run nginx --image=nginx -- <arg1> <arg2> ... <argN>

# Start the nginx container using a different command and custom arguments
$ kubectl run nginx --image=nginx --command -- <cmd> <arg1> ... <argN>
```

### Options

```
      --attach[=false]: If true, wait for the Pod to start running, and then attach to the Pod as if 'kubectl attach ...' were called.  Default false, unless '-i/--interactive' is set, in which case the default is true.
      --command[=false]: If true and extra arguments are present, use them as the 'command' field in the container, rather than the 'args' field which is the default.
      --dry-run[=false]: If true, only print the object that would be sent, without sending it.
      --generator="": The name of the API generator to use.  Default is 'run/v1' if --restart=Always, otherwise the default is 'run-pod/v1'.
  -h, --help[=false]: help for run
      --hostport=-1: The host port mapping for the container port. To demonstrate a single-machine container.
      --image="": The image for the container to run.
  -l, --labels="": Labels to apply to the pod(s).
      --no-headers[=false]: When using the default output, don't print headers.
  -o, --output="": Output format. One of: json|yaml|template|templatefile|wide|jsonpath|name.
      --output-version="": Output the formatted object with the given version (default api-version).
      --overrides="": An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.
      --port=-1: The port that this container exposes.
  -r, --replicas=1: Number of replicas to create for this container. Default is 1.
      --restart="Always": The restart policy for this Pod.  Legal values [Always, OnFailure, Never].  If set to 'Always' a replication controller is created for this pod, if set to OnFailure or Never, only the Pod is created and --replicas must be 1.  Default 'Always'
  -a, --show-all[=false]: When printing, show all resources (default hide terminated pods.)
      --sort-by="": If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. 'ObjectMeta.Name'). The field in the API resource specified by this JSONPath expression must be an integer or a string.
  -i, --stdin[=false]: Keep stdin open on the container(s) in the pod, even if nothing is attached.
      --template="": Template string or path to template file to use when -o=template, -o=templatefile or -o=jsonpath.  The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview]. The jsonpath template is composed of jsonpath expressions enclosed by {} [http://releases.k8s.io/HEAD/docs/user-guide/jsonpath.md]
      --tty[=false]: Allocated a TTY for each container in the pod.  Because -t is currently shorthand for --template, -t is not supported for --tty. This shorthand is deprecated and we expect to adopt -t for --tty soon.
```

### Options inherited from parent commands

```
      --alsologtostderr[=false]: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
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

###### Auto generated by spf13/cobra at 2015-08-26 09:03:39.976311407 +0000 UTC

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubectl/kubectl_run.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
