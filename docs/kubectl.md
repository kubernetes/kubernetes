## kubectl

kubectl controls the Kubernetes cluster manager

### Commands

#### version
Print version of client and server

Usage:
```
  kubectl version [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
  -c, --client=false: Client version only (no server required)
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
  -h, --help=false: help for version
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### proxy
Run a proxy to the Kubernetes API server.

Usage:
```
  kubectl proxy [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
  -h, --help=false: help for proxy
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -p, --port=8001: The port on which to run the proxy
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging
  -w, --www="": Also serve static files from the given directory under the prefix /static

```

#### get
Display one or many resources.

Possible resources include pods (po), replication controllers (rc), services
(se), minions (mi), or events (ev).

If you specify a Go template, you can use any fields defined for the API version
you are connecting to the server with.

Examples:
  $ kubectl get pods
  <list all pods in ps output format>

  $ kubectl get replicationController 1234-56-7890-234234-456456
  <list single replication controller in ps output format>

  $ kubectl get -o json pod 1234-56-7890-234234-456456
  <list single pod in json output format>

  $ kubectl get rc,services
  <list replication controllers and services together in ps output format>

Usage:
```
  kubectl get [(-o|--output=)json|yaml|...] <resource> [<id>] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
  -h, --help=false: help for get
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --no-headers=false: When using the default output, don't print headers
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -o, --output="": Output format: json|yaml|template|templatefile
      --output-version="": Output the formatted object with the given version (default api-version)
  -l, --selector="": Selector (label query) to filter on
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
  -t, --template="": Template string or path to template file to use when -o=template or -o=templatefile.
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging
  -w, --watch=false: After listing/getting the requested object, watch for changes.
      --watch-only=false: Watch for changes to the requested object(s), without listing/getting first.

```

#### describe
Show details of a specific resource.

This command joins many API calls together to form a detailed description of a
given resource.

Usage:
```
  kubectl describe <resource> <id> [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### create
Create a resource by filename or stdin.

JSON and YAML formats are accepted.

Examples:
  $ kubectl create -f pod.json
  <create a pod using the data in pod.json>

  $ cat pod.json | kubectl create -f -
  <create a pod based on the json passed into stdin>

Usage:
```
  kubectl create -f filename [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
  -f, --filename=[]: Filename, directory, or URL to file to use to create the resource
  -h, --help=false: help for create
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### update
Update a resource by filename or stdin.

JSON and YAML formats are accepted.

Examples:
  $ kubectl update -f pod.json
  <update a pod using the data in pod.json>

  $ cat pod.json | kubectl update -f -
  <update a pod based on the json passed into stdin>
  
  $ kubectl update pods my-pod --patch='{ "apiVersion": "v1beta1", "desiredState": { "manifest": [{ "cpu": 100 }]}}'
  <update a pod by downloading it, applying the patch, then updating, requires apiVersion be specified>

Usage:
```
  kubectl update -f filename [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
  -f, --filename="": Filename or URL to file to use to update the resource
  -h, --help=false: help for update
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
      --patch="": A JSON document to override the existing resource.  The resource is downloaded, then patched with the JSON, the updated
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### delete
Delete a resource by filename, stdin, resource and id or by resources and label selector.

JSON and YAML formats are accepted.

If both a filename and command line arguments are passed, the command line
arguments are used and the filename is ignored.

Note that the delete command does NOT do resource version checks, so if someone
submits an update to a resource right when you submit a delete, their update
will be lost along with the rest of the resource.

Examples:
  $ kubectl delete -f pod.json
  <delete a pod using the type and id pod.json>

  $ cat pod.json | kubectl delete -f -
  <delete a pod based on the type and id in the json passed into stdin>

  $ kubectl delete pods,services -l name=myLabel
  <delete pods and services with label name=myLabel>

  $ kubectl delete pod 1234-56-7890-234234-456456
  <delete a pod with ID 1234-56-7890-234234-456456>

Usage:
```
  kubectl delete ([-f filename] | (<resource> [(<id> | -l <label>)] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
  -f, --filename=[]: Filename, directory, or URL to a file containing the resource to delete
  -h, --help=false: help for delete
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -l, --selector="": Selector (label query) to filter on
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config
config modifies .kubeconfig files using subcommands like "kubectl config set current-context my-context"

Usage:
```
  kubectl config <subcommand> [flags]
  kubectl config [command]

Available Commands: 
  view                                                                                       displays the specified .kubeconfig file or a merged result
  set-cluster name [server] [insecure-skip-tls-verify] [certificate-authority] [api-version] Sets a cluster entry in .kubeconfig
  set-credentials name                                                                       Sets a user entry in .kubeconfig
  set-context name                                                                           Sets a context entry in .kubeconfig
  set property-name property-value                                                           Sets an individual value in a .kubeconfig file
  unset property-name                                                                        Unsets an individual value in a .kubeconfig file
  use-context context-name                                                                   Sets the current-context in a .kubeconfig file

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --global=false: use the .kubeconfig from /home/username
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=true: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

Additional help topics: 
  kubectl version       Print version of client and server
  kubectl proxy         Run a proxy to the Kubernetes API server
  kubectl get           Display one or many resources
  kubectl describe      Show details of a specific resource
  kubectl create        Create a resource by filename or stdin
  kubectl update        Update a resource by filename or stdin
  kubectl delete        Delete a resource by filename, stdin or resource and id
  kubectl config        config modifies .kubeconfig files
  kubectl namespace     Set and view the current Kubernetes namespace
  kubectl log           Print the logs for a container in a pod.
  kubectl rollingupdate Perform a rolling update of the given ReplicationController
  kubectl resize        Set a new size for a resizable resource (currently only Replication Controllers)
  kubectl run-container Run a particular image on the cluster.
  kubectl stop          Gracefully shutdown a resource

Use "kubectl help [command]" for more information about that command.
```

#### config view
displays the specified .kubeconfig file or a merged result

Usage:
```
  kubectl config view [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for view
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=true: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --merge=false: merge together the full hierarchy of .kubeconfig files
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config set-cluster
Sets a cluster entry in .kubeconfig

		Specifying a name that already exists overwrites that cluster entry.
		

Usage:
```
  kubectl config set-cluster name [server] [insecure-skip-tls-verify] [certificate-authority] [api-version] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": api-version for the cluster entry in .kubeconfig
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": certificate-authority for the cluster entry in .kubeconfig
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for set-cluster
      --insecure-skip-tls-verify=false: insecure-skip-tls-verify for the cluster entry in .kubeconfig
      --kubeconfig="": use a particular .kubeconfig file
      --local=true: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
      --server="": server for the cluster entry in .kubeconfig
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config set-credentials
Sets a user entry in .kubeconfig

		Specifying a name that already exists overwrites that user entry.
		

Usage:
```
  kubectl config set-credentials name [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
      --auth-path="": auth-path for the user entry in .kubeconfig
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": client-certificate for the user entry in .kubeconfig
      --client-key="": client-key for the user entry in .kubeconfig
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for set-credentials
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=true: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": token for the user entry in .kubeconfig
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config set-context
Sets a context entry in .kubeconfig

		Specifying a name that already exists overwrites that context entry.
		

Usage:
```
  kubectl config set-context name [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": cluster for the context entry in .kubeconfig
      --context="": The name of the kubeconfig context to use
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for set-context
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=true: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": namespace for the context entry in .kubeconfig
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": user for the context entry in .kubeconfig
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config set
Sets an individual value in a .kubeconfig file

		property-name is a dot delimitted name where each token represents either a attribute name or a map key.  Map keys may not contain dots.
		property-value is the new value you wish to set.

		

Usage:
```
  kubectl config set property-name property-value [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for config
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=true: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config unset
Unsets an individual value in a .kubeconfig file

		property-name is a dot delimitted name where each token represents either a attribute name or a map key.  Map keys may not contain dots.
		

Usage:
```
  kubectl config unset property-name [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for config
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=true: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config use-context
Sets the current-context in a .kubeconfig file

Usage:
```
  kubectl config use-context context-name [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for config
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=true: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### namespace
Set and view the current Kubernetes namespace scope for command line requests.

A Kubernetes namespace subdivides the cluster into groups of logically related pods, services, and replication controllers.

Examples:
  $ kubectl namespace 
  Using namespace default

  $ kubectl namespace other
  Set current namespace to other

Usage:
```
  kubectl namespace [<namespace>] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### log
Print the logs for a container in a pod. If the pod has only one container, the container name is optional
Examples:
  $ kubectl log 123456-7890 ruby-container
  <returns snapshot of ruby-container logs from pod 123456-7890>

  $ kubectl log -f 123456-7890 ruby-container
  <starts streaming of ruby-container logs from pod 123456-7890>

Usage:
```
  kubectl log [-f] <pod> [<container>] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
  -f, --follow=false: Specify if the logs should be streamed.
  -h, --help=false: help for log
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### rollingupdate
Perform a rolling update of the given ReplicationController.

Replaces named controller with new controller, updating one pod at a time to use the
new PodTemplate. The new-controller.json must specify the same namespace as the
existing controller and overwrite at least one (common) label in its replicaSelector.

Examples:
$ kubectl rollingupdate frontend-v1 -f frontend-v2.json
  <update pods of frontend-v1 using new controller data in frontend-v2.json>

$ cat frontend-v2.json | kubectl rollingupdate frontend-v1 -f -
  <update pods of frontend-v1 using json data passed into stdin>

Usage:
```
  kubectl rollingupdate <old-controller-name> -f <new-controller.json> [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
  -f, --filename="": Filename or URL to file to use to create the new controller
  -h, --help=false: help for rollingupdate
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
      --poll-interval="3s": Time delay between polling controller status after update. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --timeout="5m0s": Max time to wait for a controller to update before giving up. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
      --token="": Bearer token for authentication to the API server.
      --update-period="1m0s": Time to wait between updating pods. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### resize
Set a new size for a resizable resource (currently only Replication Controllers)

Resize also allows users to specify one or more preconditions for the resize action.
The new size is specified by --replicas=<n>. You can also specify an optional precondition. 
The two currently supported options are --current-replicas or --resource-version.
If a precondition is specified, it is validated before the resize is attempted, and it is 
guaranteed that the precondition holds true when the resize is sent to the server.

Examples:
  $ kubectl resize --replicas=3 replicationcontrollers foo
  resized

  # will only execute if the current size is 2
  $ kubectl resize --current-replicas=2 --replicas=3 replicationcontrollers foo


Usage:
```
  kubectl resize [--resource-version=<version>] [--current-replicas=<count>] --replicas=<count> <resource> <id> [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --current-replicas=-1: Precondition for current size. Requires that the current size of the replication controller match this value in order to resize
  -h, --help=false: help for resize
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
      --replicas=-1: The new number desired number of replicas.  Required.
      --resource-version="": Precondition for resource version. Requires that the current resource version match this value in order to resize
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### run-container
Create and run a particular image, possibly replicated.
Creates a replication controller to manage the created container(s)

Examples:
  $ kubectl run-container nginx --image=dockerfile/nginx
  <starts a single instance of nginx>

  $ kubectl run-container nginx --image=dockerfile/nginx --replicas=5
  <starts a replicated instance of nginx>

  $ kubectl run-container nginx --image=dockerfile/nginx --dry-run
  <just print the corresponding API objects, don't actually send them to the apiserver>
  
  $ kubectl run-container nginx --image=dockerfile/nginx --overrides='{ "apiVersion": "v1beta1", "desiredState": { ... } }'
  <start a single instance of nginx, but overload the desired state with a partial set of values parsed from JSON

Usage:
```
  kubectl run-container <name> --image=<image> [--replicas=replicas] [--dry-run=<bool>] [--overrides=<inline-json>] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --dry-run=false: If true, only print the object that would be sent, don't actually do anything
      --generator="run-container/v1": The name of the api generator that you want to use.  Default 'run-container-controller-v1'
  -h, --help=false: help for run-container
      --image="": The image for the container you wish to run.
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
  -l, --labels="": Labels to apply to the pod(s) created by this call to run.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --no-headers=false: When using the default output, don't print headers
      --ns-path="": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -o, --output="": Output format: json|yaml|template|templatefile
      --output-version="": Output the formatted object with the given version (default api-version)
      --overrides="": An inline JSON override for the generated object.  If this is non-empty, it is parsed used to override the generated object.  Requires that the object supply a valid apiVersion field.
  -r, --replicas=1: Number of replicas to create for this container. Default 1
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
  -t, --template="": Template string or path to template file to use when -o=template or -o=templatefile.
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### stop
Gracefully shutdown a resource

Attempts to shutdown and delete a resource that supports graceful termination.
If the resource is resizable it will be resized to 0 before deletion.

Examples:
  $ kubectl stop replicationcontroller foo
  foo stopped


Usage:
```
  kubectl stop <resource> <id> [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="/home/username/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

