## kubectl delete

Delete resources by filenames, stdin, resources and names, or by resources and label selector

### Synopsis


Delete resources by filenames, stdin, resources and names, or by resources and label selector. 

JSON and YAML formats are accepted. Only one type of the arguments may be specified: filenames, resources and names, or resources and label selector. 

Some resources, such as pods, support graceful deletion. These resources define a default period before they are forcibly terminated (the grace period) but you may override that value with the --grace-period flag, or pass --now to set a grace-period of 1. Because these resources often represent entities in the cluster, deletion may not be acknowledged immediately. If the node hosting a pod is down or cannot reach the API server, termination may take significantly longer than the grace period. To force delete a resource,  you must pass a grace   period of 0 and specify the --force flag. 

IMPORTANT: Force deleting pods does not wait for confirmation that the pod's processes have been terminated, which can leave those processes running until the node detects the deletion and completes graceful deletion. If your processes use shared storage or talk to a remote API and depend on the name of the pod to identify themselves, force deleting those pods may result in multiple processes running on different machines using the same identification which may lead to data corruption or inconsistency. Only force delete pods when you are sure the pod is terminated, or if your application can tolerate multiple copies of the same pod running at once. Also, if you force delete pods the scheduler may place new pods on those nodes before the node has released those resources and causing those pods to be evicted immediately. 

Note that the delete command does NOT do resource version checks, so if someone submits an update to a resource right when you submit a delete, their update will be lost along with the rest of the resource.

```
kubectl delete ([-f FILENAME] | TYPE [(NAME | -l label | --all)])
```

### Examples

```
  # Delete a pod using the type and name specified in pod.json.
  kubectl delete -f ./pod.json
  
  # Delete a pod based on the type and name in the JSON passed into stdin.
  cat pod.json | kubectl delete -f -
  
  # Delete pods and services with same names "baz" and "foo"
  kubectl delete pod,service baz foo
  
  # Delete pods and services with label name=myLabel.
  kubectl delete pods,services -l name=myLabel
  
  # Delete a pod with minimal delay
  kubectl delete pod foo --now
  
  # Force delete a pod on a dead node
  kubectl delete pod foo --grace-period=0 --force
  
  # Delete all pods
  kubectl delete pods --all
```

### Options

```
      --all                    [-all] to select all the specified resources.
      --cascade                If true, cascade the deletion of the resources managed by this resource (e.g. Pods created by a ReplicationController).  Default true. (default true)
  -f, --filename stringSlice   Filename, directory, or URL to files containing the resource to delete.
      --force                  Immediate deletion of some resources may result in inconsistency or data loss and requires confirmation.
      --grace-period int       Period of time in seconds given to the resource to terminate gracefully. Ignored if negative. (default -1)
      --ignore-not-found       Treat "resource not found" as a successful delete. Defaults to "true" when --all is specified.
      --now                    If true, resources are signaled for immediate shutdown (same as --grace-period=1).
  -o, --output string          Output mode. Use "-o name" for shorter output (resource/name).
  -R, --recursive              Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory.
  -l, --selector string        Selector (label query) to filter on.
      --timeout duration       The length of time to wait before giving up on a delete, zero means determine a timeout from the size of the object
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
