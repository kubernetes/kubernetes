## kubectl rolling-update

Perform a rolling update of the given ReplicationController

### Synopsis


Perform a rolling update of the given ReplicationController. 

Replaces the specified replication controller with a new replication controller by updating one pod at a time to use the new PodTemplate. The new-controller.json must specify the same namespace as the existing replication controller and overwrite at least one (common) label in its replicaSelector. 

! http://kubernetes.io/images/docs/kubectl_rollingupdate.svg

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
      --allow-missing-template-keys   If true, ignore any errors in templates when a field or map key is missing in the template. Only applies to golang and jsonpath output formats. (default true)
      --container string              Container name which will have its image upgraded. Only relevant when --image is specified, ignored otherwise. Required when using --image on a multi-container pod
      --deployment-label-key string   The key to use to differentiate between two different controllers, default 'deployment'.  Only relevant when --image is specified, ignored otherwise (default "deployment")
      --dry-run                       If true, only print the object that would be sent, without sending it.
  -f, --filename stringSlice          Filename or URL to file to use to create the new replication controller.
      --image string                  Image to use for upgrading the replication controller. Must be distinct from the existing image (either new image or new image tag).  Can not be used with --filename/-f
      --image-pull-policy string      Explicit policy for when to pull container images. Required when --image is same as existing image, ignored otherwise.
      --no-headers                    When using the default or custom-column output format, don't print headers (default print headers).
  -o, --output string                 Output format. One of: json|yaml|wide|name|custom-columns=...|custom-columns-file=...|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See custom columns [http://kubernetes.io/docs/user-guide/kubectl-overview/#custom-columns], golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://kubernetes.io/docs/user-guide/jsonpath].
      --poll-interval duration        Time delay between polling for replication controller status after the update. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h". (default 3s)
      --rollback                      If true, this is a request to abort an existing rollout that is partially rolled out. It effectively reverses current and next and runs a rollout
      --schema-cache-dir string       If non-empty, load/store cached API schemas in this directory, default is '$HOME/.kube/schema' (default "~/.kube/schema")
  -a, --show-all                      When printing, show all resources (default hide terminated pods.)
      --show-labels                   When printing, show all labels as the last column (default hide labels column)
      --sort-by string                If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. '{.metadata.name}'). The field in the API resource specified by this JSONPath expression must be an integer or a string.
      --template string               Template string or path to template file to use when -o=go-template, -o=go-template-file. The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview].
      --timeout duration              Max time to wait for a replication controller to update before giving up. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h". (default 5m0s)
      --update-period duration        Time to wait between updating pods. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h". (default 1m0s)
      --validate                      If true, use a schema to validate the input before sending it (default true)
```

### Options inherited from parent commands

```
      --alsologtostderr                  log to standard error as well as files
      --as string                        Username to impersonate for the operation
      --certificate-authority string     Path to a cert. file for the certificate authority
      --client-certificate string        Path to a client certificate file for TLS
      --client-key string                Path to a client key file for TLS
      --cluster string                   The name of the kubeconfig cluster to use
      --context string                   The name of the kubeconfig context to use
      --insecure-skip-tls-verify         If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure
      --kubeconfig string                Path to the kubeconfig file to use for CLI requests.
      --log-backtrace-at traceLocation   when logging hits line file:N, emit a stack trace (default :0)
      --log-dir string                   If non-empty, write log files in this directory
      --logtostderr                      log to standard error instead of files
      --match-server-version             Require server version to match client version
  -n, --namespace string                 If present, the namespace scope for this CLI request
      --password string                  Password for basic authentication to the API server
      --request-timeout string           The length of time to wait before giving up on a single server request. Non-zero values should contain a corresponding time unit (e.g. 1s, 2m, 3h). A value of zero means don't timeout requests. (default "0")
  -s, --server string                    The address and port of the Kubernetes API server
      --stderrthreshold severity         logs at or above this threshold go to stderr (default 2)
      --token string                     Bearer token for authentication to the API server
      --user string                      The name of the kubeconfig user to use
      --username string                  Username for basic authentication to the API server
  -v, --v Level                          log level for V logs
      --vmodule moduleSpec               comma-separated list of pattern=N settings for file-filtered logging
```

### SEE ALSO
* [kubectl](kubectl.md)	 - kubectl controls the Kubernetes cluster manager

###### Auto generated by spf13/cobra on 21-Feb-2017
