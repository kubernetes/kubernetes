<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## kubectl expose

Take a replication controller, service or pod and expose it as a new Kubernetes Service

### Synopsis


Take a replication controller, service or pod and expose it as a new Kubernetes Service.

Looks up a replication controller, service or pod by name and uses the selector for that resource as the
selector for a new Service on the specified port. If no labels are specified, the new service will
re-use the labels from the resource it exposes.

```
kubectl expose (-f FILENAME | TYPE NAME) [--port=port] [--protocol=TCP|UDP] [--target-port=number-or-name] [--name=name] [----external-ip=external-ip-of-service] [--type=type]
```

### Examples

```
# Create a service for a replicated nginx, which serves on port 80 and connects to the containers on port 8000.
$ kubectl expose rc nginx --port=80 --target-port=8000

# Create a service for a replication controller identified by type and name specified in "nginx-controller.yaml", which serves on port 80 and connects to the containers on port 8000.
$ kubectl expose -f nginx-controller.yaml --port=80 --target-port=8000

# Create a service for a pod valid-pod, which serves on port 444 with the name "frontend"
$ kubectl expose pod valid-pod --port=444 --name=frontend

# Create a second service based on the above service, exposing the container port 8443 as port 443 with the name "nginx-https"
$ kubectl expose service nginx --port=443 --target-port=8443 --name=nginx-https

# Create a service for a replicated streaming application on port 4100 balancing UDP traffic and named 'video-stream'.
$ kubectl expose rc streamer --port=4100 --protocol=udp --name=video-stream
```

### Options

```
      --container-port="": Synonym for --target-port
      --dry-run[=false]: If true, only print the object that would be sent, without creating it.
      --external-ip="": External IP address to set for the service. The service can be accessed by this IP in addition to its generated service IP.
  -f, --filename=[]: Filename, directory, or URL to a file identifying the resource to expose a service
      --generator="service/v2": The name of the API generator to use. There are 2 generators: 'service/v1' and 'service/v2'. The only difference between them is that service port in v1 is named 'default', while it is left unnamed in v2. Default is 'service/v2'.
  -l, --labels="": Labels to apply to the service created by this call.
      --load-balancer-ip="": IP to assign to to the Load Balancer. If empty, an ephemeral IP will be created and used(cloud-provider specific).
      --name="": The name for the newly created object.
      --no-headers[=false]: When using the default output, don't print headers.
  -o, --output="": Output format. One of: json|yaml|wide|name|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://releases.k8s.io/v1.1.0/docs/user-guide/jsonpath.md].
      --output-version="": Output the formatted object with the given version (default api-version).
      --overrides="": An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.
      --port=-1: The port that the service should serve on. Copied from the resource being exposed, if unspecified
      --protocol="TCP": The network protocol for the service to be created. Default is 'tcp'.
      --selector="": A label selector to use for this service. If empty (the default) infer the selector from the replication controller.
      --session-affinity="": If non-empty, set the session affinity for the service to this; legal values: 'None', 'ClientIP'
  -a, --show-all[=false]: When printing, show all resources (default hide terminated pods.)
      --sort-by="": If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. 'ObjectMeta.Name'). The field in the API resource specified by this JSONPath expression must be an integer or a string.
      --target-port="": Name or number for the port on the container that the service should direct traffic to. Optional.
      --template="": Template string or path to template file to use when -o=go-template, -o=go-template-file. The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview].
      --type="": Type for this service: ClusterIP, NodePort, or LoadBalancer. Default is 'ClusterIP'.
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

###### Auto generated by spf13/cobra at 2015-09-25 23:39:47.895322301 +0000 UTC



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubectl/kubectl_expose.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
