<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## kubectl create configmap

Create a configMap from a local file, directory or literal value.

### Synopsis


Create a configmap based on a file, directory, or specified literal value.

A single configmap may package one or more key/value pairs.

When creating a configmap based on a file, the key will default to the basename of the file, and the value will
default to the file content.  If the basename is an invalid key, you may specify an alternate key.

When creating a configmap based on a directory, each file whose basename is a valid key in the directory will be
packaged into the configmap.  Any directory entries except regular files are ignored (e.g. subdirectories,
symlinks, devices, pipes, etc).


```
kubectl create configmap NAME [--type=string] [--from-file=[key=]source] [--from-literal=key1=value1] [--dry-run]
```

### Examples

```
  # Create a new configmap named my-config with keys for each file in folder bar
  kubectl create configmap generic my-config --from-file=path/to/bar

  # Create a new configmap named my-config with specified keys instead of names on disk
  kubectl create configmap generic my-config --from-file=ssh-privatekey=~/.ssh/id_rsa --from-file=ssh-publickey=~/.ssh/id_rsa.pub

  # Create a new configMap named my-config with key1=config1 and key2=config2
  kubectl create configmap generic my-config --from-literal=key1=config1 --from-literal=key2=config2
```

### Options

```
      --dry-run[=false]: If true, only print the object that would be sent, without sending it.
      --from-file=[]: Key files can be specified using their file path, in which case a default name will be given to them, or optionally with a name and file path, in which case the given name will be used.  Specifying a directory will iterate each named file in the directory that is a valid configmap key.
      --from-literal=[]: Specify a key and literal value to insert in configmap (i.e. mykey=somevalue)
      --generator="configmap/v1": The name of the API generator to use.
      --no-headers[=false]: When using the default output, don't print headers.
  -o, --output="": Output format. One of: json|yaml|wide|name|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://releases.k8s.io/release-1.2/docs/user-guide/jsonpath.md].
      --output-version="": Output the formatted object with the given group version (for ex: 'extensions/v1beta1').
      --save-config[=false]: If true, the configuration of current object will be saved in its annotation. This is useful when you want to perform kubectl apply on this object in the future.
      --schema-cache-dir="~/.kube/schema": If non-empty, load/store cached API schemas in this directory, default is '$HOME/.kube/schema'
  -a, --show-all[=false]: When printing, show all resources (default hide terminated pods.)
      --show-labels[=false]: When printing, show all labels as the last column (default hide labels column)
      --sort-by="": If non-empty, sort list types using this field specification.  The field specification is expressed as a JSONPath expression (e.g. '{.metadata.name}'). The field in the API resource specified by this JSONPath expression must be an integer or a string.
      --template="": Template string or path to template file to use when -o=go-template, -o=go-template-file. The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview].
      --validate[=true]: If true, use a schema to validate the input before sending it
```

### Options inherited from parent commands

```
      --alsologtostderr[=false]: log to standard error as well as files
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client certificate file for TLS.
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

* [kubectl create](kubectl_create.md)	 - Create a resource by filename or stdin

###### Auto generated by spf13/cobra on 11-Mar-2016



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubectl/kubectl_create_configmap.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
