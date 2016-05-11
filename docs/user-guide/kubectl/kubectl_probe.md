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

## kubectl probe

Update a probe on a pod template

### Synopsis



Set or remove a liveness or readiness probe from a pod or pod template

Each container in a pod may define one or more probes that are used for general health
checking. A liveness probe is checked periodically to ensure the container is still healthy:
if the probe fails, the container is restarted. Readiness probes set or clear the ready
flag for each container, which controls whether the container's ports are included in the list
of endpoints for a service and whether a deployment can proceed. A readiness check should
indicate when your container is ready to accept incoming traffic or begin handling work.
Setting both liveness and readiness probes for each container is highly recommended.

The three probe types are:

1. Open a TCP socket on the pod IP
2. Perform an HTTP GET against a URL on a container that must return 200 OK
3. Run a command in the container that must return exit code 0

Containers that take a variable amount of time to start should set generous
initial-delay-seconds values, otherwise as your application evolves you may suddenly begin
to fail.

```
kubectl probe RESOURCE/NAME --readiness|--liveness (--get-url=URL|--open-tcp=PORT|-- CMD)
```

### Examples

```
  # Clear both readiness and liveness probes off all containers
  kubectl probe dc/registry --remove --readiness --liveness

  # Set an exec action as a liveness probe to run 'echo ok'
  kubectl probe dc/registry --liveness -- echo ok

  # Set a readiness probe to try to open a TCP socket on 3306
  kubectl probe rc/mysql --readiness --open-tcp=3306

  # Set an HTTP readiness probe for port 8080 and path /healthz over HTTP on the pod IP
  kubectl probe dc/webapp --readiness --get-url=http://:8080/healthz

  # Set an HTTP readiness probe over HTTPS on 127.0.0.1 for a hostNetwork pod
  kubectl probe dc/router --readiness --get-url=https://127.0.0.1:1936/stats

  # Set only the initial-delay-seconds field on all deployments
  kubectl probe dc --all --readiness --initial-delay-seconds=30
```

### Options

```
      --all[=false]: Select all resources in the namespace of the specified resource types
  -c, --containers="*": The names of containers in the selected pod templates to change - may use wildcards
      --failure-threshold=0: The number of failures before the probe is considered to have failed
  -f, --filename=[]: Filename, directory, or URL to file to use to edit the resource.
      --get-url="": A URL to perform an HTTP GET on (you can omit the host, have a string port, or omit the scheme.
      --initial-delay-seconds=0: The time in seconds to wait before the probe begins checking
      --liveness[=false]: Set or remove a liveness probe to verify this container is running
      --no-headers[=false]: When using the default output, don't print headers.
      --open-tcp="": A port number or port name to attempt to open via TCP.
  -o, --output="": Output format. One of: json|yaml|wide|name|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://releases.k8s.io/HEAD/docs/user-guide/jsonpath.md].
      --output-version="": Output the formatted object with the given group version (for ex: 'extensions/v1beta1').
      --period-seconds=0: The time in seconds between attempts
      --readiness[=false]: Set or remove a readiness probe to indicate when this container should receive traffic
      --remove[=false]: If true, remove the specified probe(s).
  -l, --selector="": Selector (label query) to filter on
  -a, --show-all[=false]: When printing, show all resources (default hide terminated pods.)
      --show-labels[=false]: When printing, show all labels as the last column (default hide labels column)
      --sort-by="": If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. '{.metadata.name}'). The field in the API resource specified by this JSONPath expression must be an integer or a string.
      --success-threshold=0: The number of successes required before the probe is considered successful
      --template="": Template string or path to template file to use when -o=go-template, -o=go-template-file. The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview].
      --timeout-seconds=0: The time in seconds to wait before considering the probe to have failed
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

###### Auto generated by spf13/cobra on 10-May-2016

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubectl/kubectl_probe.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
