## kubectl
The ```kubectl``` command provides command line access to the kubernetes API.

### Commands

#### version
Print the version of the client and server.

Usage: 
```
  kubectl version [flags]

 Available Flags:
      --api-version="v1beta1": The version of the API to use against the server
  -a, --auth-path="/Users/bburns/.kubernetes_auth": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a certificate file for the certificate authority
  -c, --client=false: Client version only (no server required)
      --client-certificate="": Path to a client certificate for TLS.
      --client-key="": Path to a client key file for TLS.
  -h, --help=false: help for version
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --match-server-version=false: Require server version to match client version
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="/Users/bburns/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": Kubernetes apiserver to connect to


Use "kubectl help [command]" for more information about that command.
```

#### proxy
Create a local proxy to the API server

Usage: 
```
  kubectl proxy [flags]

 Available Flags:
      --api-version="v1beta1": The API version to use when talking to the server
  -a, --auth-path="/Users/bburns/.kubernetes_auth": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority=: Path to a cert. file for the certificate authority.
      --client-certificate=: Path to a client key file for TLS.
      --client-key=: Path to a client key file for TLS.
  -h, --help=false: help for proxy
      --insecure-skip-tls-verify=%!s(bool=false): If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --match-server-version=false: Require server version to match client version
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="/Users/bburns/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -p, --port=8001: The port on which to run the proxy
  -s, --server="": The address of the Kubernetes API server
      --token=: Bearer token for authentication to the API server.
      --validate=false: If true, use a schema to validate the input before sending it
  -w, --www="": Also serve static files from the given directory under the prefix /static
```

#### get
Display one or more resources
Possible resources include pods (po), replication controllers (rc), services
(se), minions (mi), or events (ev).

If you specify a Go template, you can use any fields defined for the API version
you are connecting to the server with.

Examples:
```sh
  $ kubectl get pods
  <list all pods in ps output format>

  $ kubectl get replicationController 1234-56-7890-234234-456456
  <list single replication controller in ps output format>

  $ kubectl get -o json pod 1234-56-7890-234234-456456
  <list single pod in json output format>
```
Usage: 
```
  kubectl get [(-o|--output=)json|yaml|...] <resource> [<id>] [flags]

 Available Flags:
      --api-version="v1beta1": The API version to use when talking to the server
  -a, --auth-path="/Users/bburns/.kubernetes_auth": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority=: Path to a cert. file for the certificate authority.
      --client-certificate=: Path to a client key file for TLS.
      --client-key=: Path to a client key file for TLS.
  -h, --help=false: help for get
      --insecure-skip-tls-verify=%!s(bool=false): If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --match-server-version=false: Require server version to match client version
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --no-headers=false: When using the default output, don't print headers
      --ns-path="/Users/bburns/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -o, --output="": Output format: json|yaml|template|templatefile
      --output-version="": Output the formatted object with the given version (default api-version)
  -l, --selector="": Selector (label query) to filter on
  -s, --server="": The address of the Kubernetes API server
  -t, --template="": Template string or path to template file to use when --output=template or --output=templatefile
      --token=: Bearer token for authentication to the API server.
      --validate=false: If true, use a schema to validate the input before sending it
  -w, --watch=false: After listing/getting the requested object, watch for changes.
      --watch-only=false: Watch for changes to the requseted object(s), without listing/getting first.
```

#### describe
Show details of a specific resource.

This command joins many API calls together to form a detailed description of a
given resource.

Usage: 
```
  kubectl describe <resource> <id> [flags]

 Available Flags:
      --api-version="v1beta1": The API version to use when talking to the server
  -a, --auth-path="/Users/bburns/.kubernetes_auth": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority=: Path to a cert. file for the certificate authority.
      --client-certificate=: Path to a client key file for TLS.
      --client-key=: Path to a client key file for TLS.
  -h, --help=false: help for describe
      --insecure-skip-tls-verify=%!s(bool=false): If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --match-server-version=false: Require server version to match client version
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="/Users/bburns/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --token=: Bearer token for authentication to the API server.
      --validate=false: If true, use a schema to validate the input before sending it
```

### run-container
Easily run one or more replicas of an image.

Creates a replica controller running the specified image, with one or more replicas.

Examples:
```sh
  $ kubectl run-container nginx --image=dockerfile/nginx
  <starts a single instance of nginx>

  $ kubectl run-container nginx --image=dockerfile/nginx --replicas=5
  <starts a replicated instance of nginx>
  
  $ kubectl run-container nginx --image=dockerfile/nginx --dry-run
  <just print the corresponding API objects, don't actually send them to the apiserver>

Usage: 
  kubectl run-container <name> --image=<image> [--replicas=replicas] [--dry-run=<bool>] [flags]

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
      --generator="run-container-controller-v1": The name of the api generator that you want to use.  Default 'run-container-controller-v1'
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
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --no-headers=false: When using the default output, don't print headers
      --ns-path="/Users/bburns/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -o, --output="": Output format: json|yaml|template|templatefile
      --output-version="": Output the formatted object with the given version (default api-version)
  -r, --replicas=1: Number of replicas to create for this container. Default 1
  -s, --server="": The address of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
  -t, --template="": Template string or path to template file to use when -o=template or -o=templatefile.
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging  $ kubectl run nginx dockerfile/nginx
  <starts a single instance of nginx>
```


#### create
Create a resource by filename or stdin.

JSON and YAML formats are accepted.

Examples:
```sh
  $ kubectl create -f pod.json
  <create a pod using the data in pod.json>

  $ cat pod.json | kubectl create -f -
  <create a pod based on the json passed into stdin>
```
Usage: 
```
  kubectl create -f filename [flags]

 Available Flags:
      --api-version="v1beta1": The API version to use when talking to the server
  -a, --auth-path="/Users/bburns/.kubernetes_auth": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority=: Path to a cert. file for the certificate authority.
      --client-certificate=: Path to a client key file for TLS.
      --client-key=: Path to a client key file for TLS.
  -f, --filename="": Filename or URL to file to use to create the resource
  -h, --help=false: help for create
      --insecure-skip-tls-verify=%!s(bool=false): If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --match-server-version=false: Require server version to match client version
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="/Users/bburns/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --token=: Bearer token for authentication to the API server.
      --validate=false: If true, use a schema to validate the input before sending it
```

#### createall
Create all resources contained in JSON file specified in a directory, filename or stdin

JSON and YAML formats are accepted.

Examples:
```sh
  $ kubectl createall -d configs/
  <creates all resources listed in JSON or YAML files, found recursively under the configs directory>

  $ kubectl createall -f config.json
  <creates all resources listed in config.json>

  $ cat config.json | kubectl apply -f -
  <creates all resources listed in config.json>
```
Usage: 
```
  kubectl createall [-d directory] [-f filename] [flags]

 Available Flags:
      --api-version="v1beta1": The API version to use when talking to the server
  -a, --auth-path="/Users/bburns/.kubernetes_auth": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority=: Path to a cert. file for the certificate authority.
      --client-certificate=: Path to a client key file for TLS.
      --client-key=: Path to a client key file for TLS.
  -d, --directory="": Directory of JSON or YAML files to use to update the resource
  -f, --filename="": Filename or URL to file to use to update the resource
  -h, --help=false: help for createall
      --insecure-skip-tls-verify=%!s(bool=false): If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --match-server-version=false: Require server version to match client version
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="/Users/bburns/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --token=: Bearer token for authentication to the API server.
      --validate=false: If true, use a schema to validate the input before sending it
```

#### update
Update a resource by filename or stdin.

JSON and YAML formats are accepted.

Examples:
```sh
  $ kubectl update -f pod.json
  <update a pod using the data in pod.json>

  $ cat pod.json | kubectl update -f -
  <update a pod based on the json passed into stdin>
```

Usage: 
```
  kubectl update -f filename [flags]

 Available Flags:
      --api-version="v1beta1": The API version to use when talking to the server
  -a, --auth-path="/Users/bburns/.kubernetes_auth": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority=: Path to a cert. file for the certificate authority.
      --client-certificate=: Path to a client key file for TLS.
      --client-key=: Path to a client key file for TLS.
  -f, --filename="": Filename or URL to file to use to update the resource
  -h, --help=false: help for update
      --insecure-skip-tls-verify=%!s(bool=false): If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --match-server-version=false: Require server version to match client version
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="/Users/bburns/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --token=: Bearer token for authentication to the API server.
      --validate=false: If true, use a schema to validate the input before sending it
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
```sh
  $ kubectl delete -f pod.json
  <delete a pod using the type and id pod.json>

  $ cat pod.json | kubectl delete -f -
  <delete a pod based on the type and id in the json passed into stdin>

  $ kubectl delete pods,services -l name=myLabel
  <delete pods and services with label name=myLabel>

  $ kubectl delete pod 1234-56-7890-234234-456456
  <delete a pod with ID 1234-56-7890-234234-456456>
```

Usage: 
```
  kubectl delete ([-f filename] | (<resource> [(<id> | -l <label>)] [flags]

 Available Flags:
      --api-version="v1beta1": The API version to use when talking to the server
  -a, --auth-path="/Users/bburns/.kubernetes_auth": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority=: Path to a cert. file for the certificate authority.
      --client-certificate=: Path to a client key file for TLS.
      --client-key=: Path to a client key file for TLS.
  -f, --filename="": Filename or URL to file to use to delete the resource
  -h, --help=false: help for delete
      --insecure-skip-tls-verify=%!s(bool=false): If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --match-server-version=false: Require server version to match client version
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="/Users/bburns/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -l, --selector="": Selector (label query) to filter on
  -s, --server="": The address of the Kubernetes API server
      --token=: Bearer token for authentication to the API server.
      --validate=false: If true, use a schema to validate the input before sending it
```

#### logs
Print the logs for a container in a pod. If the pod has only one container, the container name is optional
Examples:
```sh
  $ kubectl log 123456-7890 ruby-container
  <returns snapshot of ruby-container logs from pod 123456-7890>

  $ kubectl log -f 123456-7890 ruby-container
  <starts streaming of ruby-container logs from pod 123456-7890>
```
Usage: 
```
  kubectl log [-f] <pod> [<container>] [flags]

 Available Flags:
      --api-version="v1beta1": The API version to use when talking to the server
  -a, --auth-path="/Users/bburns/.kubernetes_auth": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority=: Path to a cert. file for the certificate authority.
      --client-certificate=: Path to a client key file for TLS.
      --client-key=: Path to a client key file for TLS.
  -f, --follow=false: Specify if the logs should be streamed.
  -h, --help=false: help for log
      --insecure-skip-tls-verify=%!s(bool=false): If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --match-server-version=false: Require server version to match client version
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="/Users/bburns/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --token=: Bearer token for authentication to the API server.
      --validate=false: If true, use a schema to validate the input before sending it
```

### Usage
```
Usage: 
  kubectl [flags]
  kubectl [command]

Available Commands: 
  version                                                   Print version of client and server
  proxy                                                     Run a proxy to the Kubernetes API server
  get [(-o|--output=)json|yaml|...] <resource> [<id>]       Display one or many resources
  describe <resource> <id>                                  Show details of a specific resource
  create -f filename                                        Create a resource by filename or stdin
  createall [-d directory] [-f filename]                    Create all resources specified in a directory, filename or stdin
  update -f filename                                        Update a resource by filename or stdin
  delete ([-f filename] | (<resource> [(<id> | -l <label>)] Delete a resource by filename, stdin or resource and id
  namespace [<namespace>]                                   Set and view the current Kubernetes namespace
  log [-f] <pod> [<container>]                              Print the logs for a container in a pod.
  help [command]                                            Help about any command

 Available Flags:
      --api-version="v1beta1": The API version to use when talking to the server
  -a, --auth-path="/Users/bburns/.kubernetes_auth": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority=: Path to a cert. file for the certificate authority.
      --client-certificate=: Path to a client key file for TLS.
      --client-key=: Path to a client key file for TLS.
  -h, --help=false: help for kubectl
      --insecure-skip-tls-verify=%!s(bool=false): If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --match-server-version=false: Require server version to match client version
  -n, --namespace="": If present, the namespace scope for this CLI request.
      --ns-path="/Users/bburns/.kubernetes_ns": Path to the namespace info file that holds the namespace context to use for CLI requests.
  -s, --server="": The address of the Kubernetes API server
      --token=: Bearer token for authentication to the API server.
      --validate=false: If true, use a schema to validate the input before sending it
```


## kubecfg is deprecated.  Please use kubectl!
## kubecfg command line interface
The `kubecfg` command line tools is used to interact with the Kubernetes HTTP API.

* [ReplicationController Commands](#replication-controller-commands)
* [RESTful Commands](#restful-commands)
* [Complete Details](#details)
  * [Usage](#usage)
  * [Options](#options)

### Replication Controller Commands

#### Run
```
kubecfg [options] run <image> <replicas> <controller-name>
```

Creates a Kubernetes ReplicaController object.

* `[options]` are described in the [Options](#options) section.
* `<image>` is the Docker image to use.
* `<replicas>` is the number of replicas of the container to create.
* `<controller-name>` is the name to assign to this new ReplicaController.

##### Example

```
kubecfg -p 8080:80 run dockerfile/nginx 2 myNginxController
```

#### Resize
```
kubecfg [options] resize <controller-name> <new-size>
```

Changes the desired number of replicas, causing replicas to be created or deleted.

* `[options]` are described in the [Options](#options) section.


##### Example
```
kubecfg resize myNginxController 3
```

#### Stop
```
kubecfg [options] stop <controller-name>
```

Stops a controller by setting its desired size to zero. Syntactic sugar on top of resize.

* `[options]` are described in the [Options](#options) section.

#### Remove
```
kubecfg [options] rm <controller-name>
```

Delete a replication controller. The desired size of the controller must be zero, by
calling either `kubecfg resize <controller-name> 0` or `kubecfg stop <controller-name>`.

* `[options]` are described in the [Options](#options) section.

### RESTful Commands
Kubecfg also supports raw access to the basic restful requests. There are four different resources you can acccess:

   * `pods`
   * `replicationControllers`
   * `services`
   * `minions`

###### Common Flags

   * -yaml : output in YAML format
   * -json : output in JSON format
   * -c <config-file> : Accept a file in JSON or YAML for POST/PUT
   
#### Commands

##### get
Raw access to a RESTful GET request.

```
kubecfg [options] get pods/pod-abc-123
```

##### list
Raw access to a RESTful LIST request.

```
kubecfg [options] list pods
```

##### create
Raw access to a RESTful POST request.

```
kubecfg <-c some/body.[json|yaml]> [options] create pods
```

##### update
Raw access to a RESTful PUT request.

```
kubecfg <-c some/body.[json|yaml]> [options] update pods/pod-abc-123
```

##### delete
Raw access to a RESTful DELETE request.

```
kubecfg [options] delete pods/pod-abc-123
```


### Details

#### Usage
```
kubecfg -h [-c config/file.json] [-p :,..., :] <method>

  Kubernetes REST API:
  kubecfg [OPTIONS] get|list|create|delete|update <minions|pods|replicationControllers|services>[/<id>]

  Manage replication controllers:
  kubecfg [OPTIONS] stop|rm|rollingupdate <controller>
  kubecfg [OPTIONS] run <image> <replicas> <controller>
  kubecfg [OPTIONS] resize <controller> <replicas>
```

#### Options

* `-V=true|false`: Print the version number.
* `-alsologtostderr=true|false`: log to standard error as well as files
* `-auth="/path/to/.kubernetes_auth"`: Path to the auth info file. Only used if doing https.
* `-c="/path/to/config_file"`: Path to the config file.
* `-h=""`: The host to connect to.
* `-json=true|false`: If true, print raw JSON for responses
* `-l=""`: Selector (label query) to use for listing
* `-log_backtrace_at=:0`: when logging hits line file:N, emit a stack trace
* `-log_dir=""`: If non-empty, write log files in this directory
* `-log_flush_frequency=5s`: Maximum number of seconds between log flushes
* `-logtostderr=true|false`: log to standard error instead of files
* `-p=""`: The port spec, comma-separated list of `<external>:<internal>,...`
* `-proxy=true|false`: If true, run a proxy to the API server
* `-s=-1`: If positive, create and run a corresponding service on this port, only used with 'run'
* `-stderrthreshold=0`: logs at or above this threshold go to stderr
* `-template=""`: If present, parse this string as a golang template and use it for output printing
* `-template_file=""`: If present, load this file as a golang template and use it for output printing
* `-u=1m0s`: Update interval period
* `-v=0`: log level for V logs. See [Logging Conventions](devel/logging.md) for details
* `-verbose=true|false`: If true, print extra information
* `-vmodule=""`: comma-separated list of pattern=N settings for file-filtered logging
* `-www=""`: If -proxy is true, use this directory to serve static files
* `-yaml=true|false`: If true, print raw YAML for responses
