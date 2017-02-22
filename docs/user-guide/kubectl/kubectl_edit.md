## kubectl edit

Edit a resource on the server

### Synopsis


Edit a resource from the default editor. 

The edit command allows you to directly edit any API resource you can retrieve via the command line tools. It will open the editor defined by your KUBE _EDITOR, or EDITOR environment variables, or fall back to 'vi' for Linux or 'notepad' for Windows. You can edit multiple objects, although changes are applied one at a time. The command accepts filenames as well as command line arguments, although the files you point to must be previously saved versions of resources. 

Editing is done with the API version used to fetch the resource. To edit using a specific API version, fully-qualify the resource, version, and group. 

The default format is YAML. To edit in JSON, specify "-o json". 

The flag --windows-line-endings can be used to force Windows line endings, otherwise the default for your operating system will be used. 

In the event an error occurs while updating, a temporary file will be created on disk that contains your unapplied changes. The most common error when updating a resource is another editor changing the resource on the server. When this occurs, you will have to apply your changes to the newer version of the resource, or update your temporary saved copy to include the latest resource version.

```
kubectl edit (RESOURCE/NAME | -f FILENAME)
```

### Examples

```
  # Edit the service named 'docker-registry':
  kubectl edit svc/docker-registry
  
  # Use an alternative editor
  KUBE_EDITOR="nano" kubectl edit svc/docker-registry
  
  # Edit the job 'myjob' in JSON using the v1 API format:
  kubectl edit job.v1.batch/myjob -o json
```

### Options

```
  -f, --filename stringSlice      Filename, directory, or URL to files to use to edit the resource
  -o, --output string             Output format. One of: yaml|json. (default "yaml")
      --record                    Record current kubectl command in the resource annotation. If set to false, do not record the command. If set to true, record the command. If not set, default to updating the existing annotation value only if one already exists.
  -R, --recursive                 Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory.
      --save-config               If true, the configuration of current object will be saved in its annotation. This is useful when you want to perform kubectl apply on this object in the future.
      --schema-cache-dir string   If non-empty, load/store cached API schemas in this directory, default is '$HOME/.kube/schema' (default "~/.kube/schema")
      --validate                  If true, use a schema to validate the input before sending it (default true)
      --windows-line-endings      Use Windows line-endings (default Unix line-endings)
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
