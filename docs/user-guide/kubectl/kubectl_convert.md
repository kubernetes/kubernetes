<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<<<<<<< f2a53e3181b2a31ebfb61dec691e7f454189ac14
<<<<<<< 3bc17a26d597dbbcd6e6437e1f45f9f65f8240cf
<<<<<<< 5470488aa560ab70b5e5240b7ad2f917a7a0251f
=======
>>>>>>> Merge pull request #22410 from nikhiljindal/apiReferenceFlag
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

<<<<<<< 3bc17a26d597dbbcd6e6437e1f45f9f65f8240cf
<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.2/docs/user-guide/kubectl/kubectl_convert.md).

=======
>>>>>>> Merge pull request #22410 from nikhiljindal/apiReferenceFlag
Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->
<<<<<<< 3bc17a26d597dbbcd6e6437e1f45f9f65f8240cf
=======
>>>>>>> Versioning docs and examples for release-1.2.
=======
>>>>>>> Merge pull request #22410 from nikhiljindal/apiReferenceFlag
=======
>>>>>>> Versionize docs after cherry-picks

<!-- END MUNGE: UNVERSIONED_WARNING -->

## kubectl convert

Convert config files between different API versions

### Synopsis


Convert config files between different API versions. Both YAML
and JSON formats are accepted.

The command takes filename, directory, or URL as input, and convert it into format
of version specified by --output-version flag. If target version is not specified or
not supported, convert to latest version.

The default output will be printed to stdout in YAML format. One can use -o option
to change to output destination.


```
kubectl convert -f FILENAME
```

### Examples

```
# Convert 'pod.yaml' to latest version and print to stdout.
kubectl convert -f pod.yaml

# Convert the live state of the resource specified by 'pod.yaml' to the latest version
# and print to stdout in json format.
kubectl convert -f pod.yaml --local -o json

# Convert all files under current directory to latest version and create them all.
kubectl convert -f . | kubectl create -f -

```

### Options

```
  -f, --filename=[]: Filename, directory, or URL to file to need to get converted.
      --include-extended-apis[=true]: If true, include definitions of new APIs via calls to the API server. [default true]
      --local[=true]: If true, convert will NOT try to contact api-server but run locally.
      --no-headers[=false]: When using the default output, don't print headers.
<<<<<<< 5470488aa560ab70b5e5240b7ad2f917a7a0251f
  -o, --output="": Output format. One of: json|yaml|wide|name|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://releases.k8s.io/HEAD/docs/user-guide/jsonpath.md].
      --output-version="": Output the formatted object with the given group version (for ex: 'extensions/v1beta1').
  -R, --recursive[=false]: If true, process directory recursively.
=======
  -o, --output="": Output format. One of: json|yaml|wide|name|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://releases.k8s.io/release-1.2/docs/user-guide/jsonpath.md].
<<<<<<< 3bc17a26d597dbbcd6e6437e1f45f9f65f8240cf
      --output-version="": Output the formatted object with the given version (default api-version).
>>>>>>> Versioning docs and examples for release-1.2.
=======
      --output-version="": Output the formatted object with the given group version (for ex: 'extensions/v1beta1').
>>>>>>> Merge pull request #22410 from nikhiljindal/apiReferenceFlag
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

* [kubectl](kubectl.md)	 - kubectl controls the Kubernetes cluster manager

<<<<<<< 3bc17a26d597dbbcd6e6437e1f45f9f65f8240cf
###### Auto generated by spf13/cobra on 30-Mar-2016



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->

=======
###### Auto generated by spf13/cobra on 11-Mar-2016
>>>>>>> Merge pull request #22410 from nikhiljindal/apiReferenceFlag



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/kubectl/kubectl_convert.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
