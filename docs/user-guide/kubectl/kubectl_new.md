<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/kubectl/kubectl_new.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## kubectl new

Create a new object of the given type by filling in a template

### Synopsis


Create a new object of the given type using a template.

The new command pulls down an example for the given object type, allows
you to edit the filled in example, and then submits it for creation.  The
editing functionality functions similarly to the 'edit' command -- you
can control the editor used, as well as the format and version, using
the same flags and environment variables.

By default, before submission your new object will be validated.  If an
invalid object is encountered, you will be returned to the editor with the
invalid part indicated with comments.  Use the '--validate=false' in order
to skip local validation.

You can specify namespace, name, and labels (in 'key1=value1,key2=value2' form)
from the command line using the '--namespace', '--name', and '--labels' flags.

You can use the '--dry-run' flag to just output the example to standard out
without launching an editor or submitting the object for creation.


```
kubectl new TYPE [--name NAME] [--labels key1=val,key2=val]
```

### Examples

```
# Create a new pod named 'website':
  $ kubectl new pod --name website --labels end=front,app=site

  # Don't validate the object before creation
  $ kubectl new namespace --validate=false --name my-ns

  # Don't edit or create a new object -- just output the example
  $ kubectl new namespace --name my-ns --dry-run
```

### Options

```
      --dry-run[=false]: If true, only print the example object, without editing or creating it.
      --labels="": A comma-separated list of labels in key=value form to apply to the created object
      --name="": The name of the new object
  -o, --output="yaml": Output format. One of: 'yaml' or 'json'.
      --output-version="": Output the formatted object with the given version (default api-version).
      --schema-cache-dir="~/.kube/schema": If non-empty, load/store cached API schemas in this directory, default is '$HOME/.kube/schema'
      --validate[=true]: If true, use a schema to validate the input before sending it
```

### Options inherited from parent commands

```
      --alsologtostderr[=false]: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
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

* [kubectl](kubectl.md)	 - kubectl controls the Kubernetes cluster manager

###### Auto generated by spf13/cobra at 2015-10-01 17:45:42.444817341 +0000 UTC

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubectl/kubectl_new.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
