## kubectl create

Create a resource by filename or stdin

### Synopsis


Create a resource by filename or stdin. 

JSON and YAML formats are accepted.

```
kubectl create -f FILENAME
```

### Examples

```
  # Create a pod using the data in pod.json.
  kubectl create -f ./pod.json
  
  # Create a pod based on the JSON passed into stdin.
  cat pod.json | kubectl create -f -
  
  # Edit the data in docker-registry.yaml in JSON using the v1 API format then create the resource using the edited data.
  kubectl create -f docker-registry.yaml --edit --output-version=v1 -o json
```

### Options

```
      --allow-missing-template-keys   If true, ignore any errors in templates when a field or map key is missing in the template. Only applies to golang and jsonpath output formats. (default true)
      --dry-run                       If true, only print the object that would be sent, without sending it.
      --edit                          Edit the API resource before creating
  -f, --filename stringSlice          Filename, directory, or URL to files to use to create the resource
      --no-headers                    When using the default or custom-column output format, don't print headers (default print headers).
  -o, --output string                 Output format. One of: json|yaml|wide|name|custom-columns=...|custom-columns-file=...|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See custom columns [http://kubernetes.io/docs/user-guide/kubectl-overview/#custom-columns], golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://kubernetes.io/docs/user-guide/jsonpath].
      --record                        Record current kubectl command in the resource annotation. If set to false, do not record the command. If set to true, record the command. If not set, default to updating the existing annotation value only if one already exists.
  -R, --recursive                     Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory.
      --save-config                   If true, the configuration of current object will be saved in its annotation. This is useful when you want to perform kubectl apply on this object in the future.
      --schema-cache-dir string       If non-empty, load/store cached API schemas in this directory, default is '$HOME/.kube/schema' (default "~/.kube/schema")
  -l, --selector string               Selector (label query) to filter on, supports '=', '==', and '!='.
  -a, --show-all                      When printing, show all resources (default hide terminated pods.)
      --show-labels                   When printing, show all labels as the last column (default hide labels column)
      --sort-by string                If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. '{.metadata.name}'). The field in the API resource specified by this JSONPath expression must be an integer or a string.
      --template string               Template string or path to template file to use when -o=go-template, -o=go-template-file. The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview].
      --validate                      If true, use a schema to validate the input before sending it (default true)
      --windows-line-endings          Only relevant if --edit=true. Use Windows line-endings (default Unix line-endings)
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
* [kubectl create clusterrolebinding](kubectl_create_clusterrolebinding.md)	 - Create a ClusterRoleBinding for a particular ClusterRole
* [kubectl create configmap](kubectl_create_configmap.md)	 - Create a configmap from a local file, directory or literal value
* [kubectl create deployment](kubectl_create_deployment.md)	 - Create a deployment with the specified name.
* [kubectl create namespace](kubectl_create_namespace.md)	 - Create a namespace with the specified name
* [kubectl create poddisruptionbudget](kubectl_create_poddisruptionbudget.md)	 - Create a pod disruption budget with the specified name.
* [kubectl create quota](kubectl_create_quota.md)	 - Create a quota with the specified name.
* [kubectl create role](kubectl_create_role.md)	 - Create a role with single rule.
* [kubectl create rolebinding](kubectl_create_rolebinding.md)	 - Create a RoleBinding for a particular Role or ClusterRole
* [kubectl create secret](kubectl_create_secret.md)	 - Create a secret using specified subcommand
* [kubectl create service](kubectl_create_service.md)	 - Create a service using specified subcommand.
* [kubectl create serviceaccount](kubectl_create_serviceaccount.md)	 - Create a service account with the specified name

###### Auto generated by spf13/cobra on 21-Feb-2017
