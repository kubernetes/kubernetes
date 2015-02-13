## kubectl

kubectl controls the Kubernetes cluster manager

### Commands

#### version
Print the client and server version information.

Usage:
```
  kubectl version [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
  -c, --client=false: Client version only (no server required).
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
  -s, --server="": The address and port of the Kubernetes API server
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
      --api-prefix="/api/": Prefix to serve the proxied API under.
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
  -p, --port=8001: The port on which to run the proxy.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging
  -w, --www="": Also serve static files from the given directory under the specified prefix.
  -P, --www-prefix="/static/": Prefix to serve static files under, if static file directory is specified.

```

#### get
Display one or many resources.

Possible resources include pods (po), replication controllers (rc), services
(se), minions (mi), or events (ev).

By specifying the output as 'template' and providing a Go template as the value
of the --template flag, you can filter the attributes of the fetched resource(s).

Examples:

    // List all pods in ps output format.
    $ kubectl get pods

    // List a single replication controller with specified ID in ps output format.
    $ kubectl get replicationController 1234-56-7890-234234-456456

    // List a single pod in JSON output format.
    $ kubectl get -o json pod 1234-56-7890-234234-456456

    // Return only the status value of the specified pod.
    $ kubectl get -o template pod 1234-56-7890-234234-456456 --template={{.currentState.status}}

    // List all replication controllers and services together in ps output format.
    $ kubectl get rc,services

Usage:
```
  kubectl get [(-o|--output=)json|yaml|template|...] <resource> [<id>] [flags]

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
      --no-headers=false: When using the default output, don't print headers.
  -o, --output="": Output format. One of: json|yaml|template|templatefile.
      --output-version="": Output the formatted object with the given version (default api-version).
  -l, --selector="": Selector (label query) to filter on
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
  -t, --template="": Template string or path to template file to use when -o=template or -o=templatefile.  The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview]
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
  -s, --server="": The address and port of the Kubernetes API server
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

    // Create a pod using the data in pod.json.
    $ kubectl create -f pod.json

    // Create a pod based on the JSON passed into stdin.
    $ cat pod.json | kubectl create -f -

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
  -s, --server="": The address and port of the Kubernetes API server
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

    // Update a pod using the data in pod.json.
    $ kubectl update -f pod.json

    // Update a pod based on the JSON passed into stdin.
    $ cat pod.json | kubectl update -f -

    // Update a pod by downloading it, applying the patch, then updating. Requires apiVersion be specified.
    $ kubectl update pods my-pod --patch='{ "apiVersion": "v1beta1", "desiredState": { "manifest": [{ "cpu": 100 }]}}'

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
  -f, --filename=[]: Filename, directory, or URL to file to use to update the resource.
  -h, --help=false: help for update
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --patch="": A JSON document to override the existing resource. The resource is downloaded, patched with the JSON, then updated.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### delete
Delete a resource by filename, stdin, resource and ID, or by resources and label selector.

JSON and YAML formats are accepted.

If both a filename and command line arguments are passed, the command line
arguments are used and the filename is ignored.

Note that the delete command does NOT do resource version checks, so if someone
submits an update to a resource right when you submit a delete, their update
will be lost along with the rest of the resource.

Examples:

    // Delete a pod using the type and ID specified in pod.json.
    $ kubectl delete -f pod.json

    // Delete a pod based on the type and ID in the JSON passed into stdin.
    $ cat pod.json | kubectl delete -f -

    // Delete pods and services with label name=myLabel.
    $ kubectl delete pods,services -l name=myLabel

    // Delete a pod with ID 1234-56-7890-234234-456456.
    $ kubectl delete pod 1234-56-7890-234234-456456

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
  -l, --selector="": Selector (label query) to filter on
  -s, --server="": The address and port of the Kubernetes API server
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
  view                                                                                                                                                              displays merged .kubeconfig settings or a specified .kubeconfig file.
  set-cluster name [--server=server] [--certificate-authority=path/to/certficate/authority] [--api-version=apiversion] [--insecure-skip-tls-verify=true]            Sets a cluster entry in .kubeconfig
  set-credentials name [--auth-path=path/to/auth/file] [--client-certificate=path/to/certficate/file] [--client-key=path/to/key/file] [--token=bearer_token_string] Sets a user entry in .kubeconfig
  set-context name [--cluster=cluster-nickname] [--user=user-nickname] [--namespace=namespace]                                                                      Sets a context entry in .kubeconfig
  set property-name property-value                                                                                                                                  Sets an individual value in a .kubeconfig file
  unset property-name                                                                                                                                               Unsets an individual value in a .kubeconfig file
  use-context context-name                                                                                                                                          Sets the current-context in a .kubeconfig file

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --envvar=false: use the .kubeconfig from $KUBECONFIG
      --global=false: use the .kubeconfig from /home/username
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=false: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

Additional help topics: 
  kubectl version       Print the client and server version information.
  kubectl proxy         Run a proxy to the Kubernetes API server
  kubectl get           Display one or many resources
  kubectl describe      Show details of a specific resource
  kubectl create        Create a resource by filename or stdin
  kubectl update        Update a resource by filename or stdin.
  kubectl delete        Delete a resource by filename, stdin, or resource and ID.
  kubectl config        config modifies .kubeconfig files
  kubectl namespace     SUPERCEDED: Set and view the current Kubernetes namespace
  kubectl log           Print the logs for a container in a pod.
  kubectl rollingupdate Perform a rolling update of the given ReplicationController.
  kubectl resize        Set a new size for a Replication Controller.
  kubectl run-container Run a particular image on the cluster.
  kubectl stop          Gracefully shut down a resource.
  kubectl expose        Take a replicated application and expose it as Kubernetes Service
  kubectl label         Update the labels on a resource

Use "kubectl help [command]" for more information about that command.
```

#### config view
displays merged .kubeconfig settings or a specified .kubeconfig file.
Examples:
  // Show merged .kubeconfig settings.
  $ kubectl config view

  // Show only local ./.kubeconfig settings
  $ kubectl config view --local

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
      --envvar=false: use the .kubeconfig from $KUBECONFIG
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for view
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=false: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --merge=true: merge together the full hierarchy of .kubeconfig files
      --namespace="": If present, the namespace scope for this CLI request.
      --no-headers=false: When using the default output, don't print headers.
  -o, --output="": Output format. One of: json|yaml|template|templatefile.
      --output-version="": Output the formatted object with the given version (default api-version).
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
  -t, --template="": Template string or path to template file to use when -o=template or -o=templatefile.  The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview]
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config set-cluster
Sets a cluster entry in .kubeconfig
	Specifying a name that already exists will merge new fields on top of existing values for those fields.
	e.g. 
		kubectl config set-cluster e2e --certificate-authority=~/.kube/e2e/.kubernetes.ca.cert
		only sets the certificate-authority field on the e2e cluster entry without touching other values.
		

Usage:
```
  kubectl config set-cluster name [--server=server] [--certificate-authority=path/to/certficate/authority] [--api-version=apiversion] [--insecure-skip-tls-verify=true] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version=: api-version for the cluster entry in .kubeconfig
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority=: certificate-authority for the cluster entry in .kubeconfig
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --envvar=false: use the .kubeconfig from $KUBECONFIG
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for set-cluster
      --insecure-skip-tls-verify=false: insecure-skip-tls-verify for the cluster entry in .kubeconfig
      --kubeconfig="": use a particular .kubeconfig file
      --local=false: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --server=: server for the cluster entry in .kubeconfig
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config set-credentials
Sets a user entry in .kubeconfig
	Specifying a name that already exists will merge new fields on top of existing values for those fields.
	e.g. 
		kubectl config set-credentials cluster-admin --client-key=~/.kube/cluster-admin/.kubecfg.key
		only sets the client-key field on the cluster-admin user entry without touching other values.
		

Usage:
```
  kubectl config set-credentials name [--auth-path=path/to/auth/file] [--client-certificate=path/to/certficate/file] [--client-key=path/to/key/file] [--token=bearer_token_string] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
      --auth-path=: auth-path for the user entry in .kubeconfig
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate=: client-certificate for the user entry in .kubeconfig
      --client-key=: client-key for the user entry in .kubeconfig
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --envvar=false: use the .kubeconfig from $KUBECONFIG
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for set-credentials
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=false: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token=: token for the user entry in .kubeconfig
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config set-context
Sets a context entry in .kubeconfig
	Specifying a name that already exists will merge new fields on top of existing values for those fields.
	e.g. 
		kubectl config set-context gce --user=cluster-admin
		only sets the user field on the gce context entry without touching other values.
		

Usage:
```
  kubectl config set-context name [--cluster=cluster-nickname] [--user=user-nickname] [--namespace=namespace] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster=: cluster for the context entry in .kubeconfig
      --context="": The name of the kubeconfig context to use
      --envvar=false: use the .kubeconfig from $KUBECONFIG
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for set-context
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=false: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace=: namespace for the context entry in .kubeconfig
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user=: user for the context entry in .kubeconfig
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config set
Sets an individual value in a .kubeconfig file

		property-name is a dot delimited name where each token represents either a attribute name or a map key.  Map keys may not contain dots.
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
      --envvar=false: use the .kubeconfig from $KUBECONFIG
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for config
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=false: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### config unset
Unsets an individual value in a .kubeconfig file

		property-name is a dot delimited name where each token represents either a attribute name or a map key.  Map keys may not contain dots.
		

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
      --envvar=false: use the .kubeconfig from $KUBECONFIG
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for config
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=false: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
  -s, --server="": The address and port of the Kubernetes API server
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
      --envvar=false: use the .kubeconfig from $KUBECONFIG
      --global=false: use the .kubeconfig from /home/username
  -h, --help=false: help for config
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": use a particular .kubeconfig file
      --local=false: use the .kubeconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### namespace
SUPERCEDED:  Set and view the current Kubernetes namespace scope for command line requests.

namespace has been superceded by the context.namespace field of .kubeconfig files.  See 'kubectl config set-context --help' for more details.


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
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### log
Print the logs for a container in a pod. If the pod has only one container, the container name is optional.

Examples:

    // Returns snapshot of ruby-container logs from pod 123456-7890.
    $ kubectl log 123456-7890 ruby-container

    // Starts streaming of ruby-container logs from pod 123456-7890.
    $ kubectl log -f 123456-7890 ruby-container

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
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### rollingupdate
Perform a rolling update of the given ReplicationController.

Replaces the specified controller with new controller, updating one pod at a time to use the
new PodTemplate. The new-controller.json must specify the same namespace as the
existing controller and overwrite at least one (common) label in its replicaSelector.

Examples:

    // Update pods of frontend-v1 using new controller data in frontend-v2.json.
    $ kubectl rollingupdate frontend-v1 -f frontend-v2.json

    // Update pods of frontend-v1 using JSON data passed into stdin.
    $ cat frontend-v2.json | kubectl rollingupdate frontend-v1 -f -

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
  -f, --filename="": Filename or URL to file to use to create the new controller.
  -h, --help=false: help for rollingupdate
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --poll-interval="3s": Time delay between polling controller status after update. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".
  -s, --server="": The address and port of the Kubernetes API server
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
Set a new size for a Replication Controller.

Resize also allows users to specify one or more preconditions for the resize action.
If --current-replicas or --resource-version is specified, it is validated before the
resize is attempted, and it is guaranteed that the precondition holds true when the
resize is sent to the server.

Examples:

    // Resize replication controller named 'foo' to 3.
    $ kubectl resize --replicas=3 replicationcontrollers foo

    // If the replication controller named foo's current size is 2, resize foo to 3.
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
      --current-replicas=-1: Precondition for current size. Requires that the current size of the replication controller match this value in order to resize.
  -h, --help=false: help for resize
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --replicas=-1: The new desired number of replicas. Required.
      --resource-version="": Precondition for resource version. Requires that the current resource version match this value in order to resize.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### run-container
Create and run a particular image, possibly replicated.
Creates a replication controller to manage the created container(s).

Examples:

    // Starts a single instance of nginx.
    $ kubectl run-container nginx --image=dockerfile/nginx

    // Starts a replicated instance of nginx.
    $ kubectl run-container nginx --image=dockerfile/nginx --replicas=5

    // Dry run. Print the corresponding API objects without creating them.
    $ kubectl run-container nginx --image=dockerfile/nginx --dry-run
  
    // Start a single instance of nginx, but overload the desired state with a partial set of values parsed from JSON.
    $ kubectl run-container nginx --image=dockerfile/nginx --overrides='{ "apiVersion": "v1beta1", "desiredState": { ... } }'

Usage:
```
  kubectl run-container <name> --image=<image> [--port=<port>] [--replicas=replicas] [--dry-run=<bool>] [--overrides=<inline-json>] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
      --dry-run=false: If true, only print the object that would be sent, without sending it.
      --generator="run-container/v1": The name of the API generator to use.  Default is 'run-container-controller/v1'.
  -h, --help=false: help for run-container
      --image="": The image for the container to run.
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
  -l, --labels="": Labels to apply to the pod(s) created by this call to run-container.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --no-headers=false: When using the default output, don't print headers.
  -o, --output="": Output format. One of: json|yaml|template|templatefile.
      --output-version="": Output the formatted object with the given version (default api-version).
      --overrides="": An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.
      --port=-1: The port that this container exposes.
  -r, --replicas=1: Number of replicas to create for this container. Default is 1.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
  -t, --template="": Template string or path to template file to use when -o=template or -o=templatefile.  The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview]
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### stop
Gracefully shut down a resource.

Attempts to shut down and delete a resource that supports graceful termination.
If the resource is resizable it will be resized to 0 before deletion.

Examples:

    // Shut down foo.
    $ kubectl stop replicationcontroller foo

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
      --namespace="": If present, the namespace scope for this CLI request.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### expose
Take a replicated application and expose it as Kubernetes Service.
		
Looks up a ReplicationController by name, and uses the selector for that replication controller
as the selector for a new Service on the specified port.

Examples:

    // Creates a service for a replicated nginx, which serves on port 80 and connects to the containers on port 8000.
    $ kubectl expose nginx --port=80 --container-port=8000

    // Create a service for a replicated streaming application on port 4100 balancing UDP traffic and named 'video-stream'.
    $ kubectl expose streamer --port=4100 --protocol=udp --service-name=video-stream

Usage:
```
  kubectl expose <name> --port=<port> [--protocol=TCP|UDP] [--container-port=<number-or-name>] [--service-name=<name>] [--public-ip=<ip>] [--create-external-load-balancer] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --container-port="": Name or number for the port on the container that the service should direct traffic to. Optional.
      --context="": The name of the kubeconfig context to use
      --create-external-load-balancer=false: If true, create an external load balancer for this service. Implementation is cloud provider dependent. Default is 'false'.
      --dry-run=false: If true, only print the object that would be sent, without creating it.
      --generator="service/v1": The name of the API generator to use.  Default is 'service/v1'.
  -h, --help=false: help for expose
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --no-headers=false: When using the default output, don't print headers.
  -o, --output="": Output format. One of: json|yaml|template|templatefile.
      --output-version="": Output the formatted object with the given version (default api-version).
      --overrides="": An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.
      --port=-1: The port that the service should serve on. Required.
      --protocol="TCP": The network protocol for the service to be created. Default is 'tcp'.
      --public-ip="": Name of a public IP address to set for the service. The service will be assigned this IP in addition to its generated service IP.
      --selector="": A label selector to use for this service. If empty (the default) infer the selector from the replication controller.
  -s, --server="": The address and port of the Kubernetes API server
      --service-name="": The name for the newly created service.
      --stderrthreshold=2: logs at or above this threshold go to stderr
  -t, --template="": Template string or path to template file to use when -o=template or -o=templatefile.  The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview]
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

#### label
Update the labels on a resource.

If --overwrite is true, then existing labels can be overwritten, otherwise attempting to overwrite a label will result in an error.
If --resource-version is specified, then updates will use this resource version, otherwise the existing resource-version will be used.

Examples:
  // Update pod 'foo' with the label 'unhealthy' and the value 'true'.
  $ kubectl label pods foo unhealthy=true

  // Update pod 'foo' with the label 'status' and the value 'unhealthy', overwriting any existing value.
  $ kubectl label --overwrite pods foo status=unhealthy
  
  // Update pod 'foo' only if the resource is unchanged from version 1.
  $ kubectl label pods foo status=unhealthy --resource-version=1
  
  // Update pod 'foo' by removing a label named 'bar' if it exists.
  // Does not require the --overwrite flag.
  $ kubectl label pods foo bar-

Usage:
```
  kubectl label [--overwrite] <resource> <name> <key-1>=<val-1> ... <key-n>=<val-n> [--resource-version=<version>] [flags]

 Available Flags:
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
  -h, --help=false: help for label
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --no-headers=false: When using the default output, don't print headers.
  -o, --output="": Output format. One of: json|yaml|template|templatefile.
      --output-version="": Output the formatted object with the given version (default api-version).
      --overwrite=false: If true, allow labels to be overwritten, otherwise reject label updates that overwrite existing labels.
      --resource-version="": If non-empty, the labels update will only succeed if this is the current resource-version for the object.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
  -t, --template="": Template string or path to template file to use when -o=template or -o=templatefile.  The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview]
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging

```

