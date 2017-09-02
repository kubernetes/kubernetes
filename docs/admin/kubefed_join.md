## kubefed join

Join a cluster to a federation

### Synopsis


Join adds a cluster to a federation. 

    Current context is assumed to be a federation API
    server. Please use the --context flag otherwise.

```
kubefed join CLUSTER_NAME --host-cluster-context=HOST_CONTEXT
```

### Examples

```
  # Join a cluster to a federation by specifying the
  # cluster name and the context name of the federation
  # control plane's host cluster. Cluster name must be
  # a valid RFC 1123 subdomain name. Cluster context
  # must be specified if the cluster name is different
  # than the cluster's context in the local kubeconfig.
  kubefed join foo --host-cluster-context=bar
```

### Options

```
      --allow-missing-template-keys          If true, ignore any errors in templates when a field or map key is missing in the template. Only applies to golang and jsonpath output formats. (default true)
      --cluster-context string               Name of the cluster's context in the local kubeconfig. Defaults to cluster name if unspecified.
      --dry-run                              If true, only print the object that would be sent, without sending it.
      --federation-system-namespace string   Namespace in the host cluster where the federation system components are installed (default "federation-system")
      --generator string                     The name of the API generator to use. (default "cluster/v1beta1")
      --host-cluster-context string          Host cluster context
      --kubeconfig string                    Path to the kubeconfig file to use for CLI requests.
      --no-headers                           When using the default or custom-column output format, don't print headers (default print headers).
      --openapi-validation                   If true, use openapi rather than swagger for validation.
  -o, --output string                        Output format. One of: json|yaml|wide|name|custom-columns=...|custom-columns-file=...|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See custom columns [http://kubernetes.io/docs/user-guide/kubectl-overview/#custom-columns], golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://kubernetes.io/docs/user-guide/jsonpath].
      --save-config                          If true, the configuration of current object will be saved in its annotation. Otherwise, the annotation will be unchanged. This flag is useful when you want to perform kubectl apply on this object in the future.
      --schema-cache-dir string              If non-empty, load/store cached API schemas in this directory, default is '$HOME/.kube/schema' (default "~/.kube/schema")
  -a, --show-all                             When printing, show all resources (default hide terminated pods.)
      --show-labels                          When printing, show all labels as the last column (default hide labels column)
      --sort-by string                       If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. '{.metadata.name}'). The field in the API resource specified by this JSONPath expression must be an integer or a string.
      --template string                      Template string or path to template file to use when -o=go-template, -o=go-template-file. The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview].
      --validate                             If true, use a schema to validate the input before sending it (default true)
```

### Options inherited from parent commands

```
      --alsologtostderr                         log to standard error as well as files
      --as string                               Username to impersonate for the operation
      --as-group stringArray                    Group to impersonate for the operation, this flag can be repeated to specify multiple groups.
      --cache-dir string                        Default HTTP cache directory (default "/usr/local/google/home/abw/.kube/http-cache")
      --certificate-authority string            Path to a cert file for the certificate authority
      --client-certificate string               Path to a client certificate file for TLS
      --client-key string                       Path to a client key file for TLS
      --cloud-provider-gce-lb-src-cidrs cidrs   CIDRS opened in GCE firewall for LB traffic proxy & health checks (default 130.211.0.0/22,35.191.0.0/16,209.85.152.0/22,209.85.204.0/22)
      --cluster string                          The name of the kubeconfig cluster to use
      --context string                          The name of the kubeconfig context to use
      --insecure-skip-tls-verify                If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure
      --log-backtrace-at traceLocation          when logging hits line file:N, emit a stack trace (default :0)
      --log-dir string                          If non-empty, write log files in this directory
      --log-flush-frequency duration            Maximum number of seconds between log flushes (default 5s)
      --logtostderr                             log to standard error instead of files (default true)
      --match-server-version                    Require server version to match client version
  -n, --namespace string                        If present, the namespace scope for this CLI request
      --password string                         Password for basic authentication to the API server
      --request-timeout string                  The length of time to wait before giving up on a single server request. Non-zero values should contain a corresponding time unit (e.g. 1s, 2m, 3h). A value of zero means don't timeout requests. (default "0")
  -s, --server string                           The address and port of the Kubernetes API server
      --stderrthreshold severity                logs at or above this threshold go to stderr (default 2)
      --token string                            Bearer token for authentication to the API server
      --user string                             The name of the kubeconfig user to use
      --username string                         Username for basic authentication to the API server
  -v, --v Level                                 log level for V logs
      --vmodule moduleSpec                      comma-separated list of pattern=N settings for file-filtered logging
```

### SEE ALSO
* [kubefed](kubefed.md)	 - kubefed controls a Kubernetes Cluster Federation

###### Auto generated by spf13/cobra on 1-Sep-2017
