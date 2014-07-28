## kubecfg command line interface
The `kubecfg` command line tools is used to interact with the Kubernetes HTTP API.

   * [ReplicationController Commands](#replication-controller-commands)
   * [RESTful Commands](#restful-commands)
   * [Complete Details](#details)

### Replication Controller Commands

#### Run
```
kubecfg [options] run <image> <replicas> <controller-name>
```

This command creates a Kuberenetes ReplicaController object which creates `<replicas>` copies of the container named by `<image>`, the newly created ReplicaController is named `<controller-name>`

##### Example

```
kubecfg -p 8080:80 run dockerfile/nginx 2 myNginxController
```

#### Resize
```
kubecfg [options] resize <controller-name> <new-size>
```

Changes the desired number of replicas, causing replicas to be created or deleted.

##### Example
```
kubecfg resize myNginxController 3
```

#### Stop
```
kubecfg [options] stop <controller-name>
```

Stops a controller by setting its desired size to zero.  Syntactic sugar on top of resize.

#### Remove
```
kubecfg [options] rm <controller-name>
```

Delete a replication controller.  Only works if the desired size of the controller is zero.

### RESTful Commands
Kubecfg also supports raw access to the basic restful requests.  There are four different resources you can acccess:

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
```
usage: kubecfg -h [-c config/file.json] [-p :,..., :] <method>

  Kubernetes REST API:
  kubecfg [OPTIONS] get|list|create|delete|update <minions|pods|replicationControllers|services>[/<id>]

  Manage replication controllers:
  kubecfg [OPTIONS] stop|rm|rollingupdate <controller>
  kubecfg [OPTIONS] run <image> <replicas> <controller>
  kubecfg [OPTIONS] resize <controller> <replicas>

  Options:
  -V=false: Print the version number.
  -alsologtostderr=false: log to standard error as well as files
  -auth="/Users/bburns/.kubernetes_auth": Path to the auth info file.  If missing, prompt the user.  Only used if doing https.
  -c="": Path to the config file.
  -h="": The host to connect to.
  -json=false: If true, print raw JSON for responses
  -l="": Selector (label query) to use for listing
  -log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
  -log_dir="": If non-empty, write log files in this directory
  -log_flush_frequency=5s: Maximum number of seconds between log flushes
  -logtostderr=false: log to standard error instead of files
  -p="": The port spec, comma-separated list of <external>:<internal>,...
  -proxy=false: If true, run a proxy to the api server
  -s=-1: If positive, create and run a corresponding service on this port, only used with 'run'
  -stderrthreshold=0: logs at or above this threshold go to stderr
  -template="": If present, parse this string as a golang template and use it for output printing
  -template_file="": If present, load this file as a golang template and use it for output printing
  -u=1m0s: Update interval period
  -v=0: log level for V logs
  -verbose=false: If true, print extra information
  -vmodule=: comma-separated list of pattern=N settings for file-filtered logging
  -www="": If -proxy is true, use this directory to serve static files
  -yaml=false: If true, print raw YAML for responses
```

