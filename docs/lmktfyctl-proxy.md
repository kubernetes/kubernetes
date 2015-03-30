## lmktfyctl proxy

Run a proxy to the LMKTFY API server

### Synopsis


Run a proxy to the LMKTFY API server. 

```
lmktfyctl proxy [--port=PORT] [--www=static-dir] [--www-prefix=prefix] [--api-prefix=prefix]
```

### Examples

```
// Run a proxy to lmktfy apiserver on port 8011, serving static content from ./local/www/
$ lmktfyctl proxy --port=8011 --www=./local/www/

// Run a proxy to lmktfy apiserver, changing the api prefix to lmktfy-api
// This makes e.g. the pods api available at localhost:8011/lmktfy-api/v1beta1/pods/
$ lmktfyctl proxy --api-prefix=lmktfy-api
```

### Options

```
      --api-prefix="/api/": Prefix to serve the proxied API under.
  -h, --help=false: help for proxy
  -p, --port=8001: The port on which to run the proxy.
  -w, --www="": Also serve static files from the given directory under the specified prefix.
  -P, --www-prefix="/static/": Prefix to serve static files under, if static file directory is specified.
```

### Options inherrited from parent commands

```
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the lmktfyconfig cluster to use
      --context="": The name of the lmktfyconfig context to use
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --lmktfyconfig="": Path to the lmktfyconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --password="": Password for basic authentication to the API server.
  -s, --server="": The address and port of the LMKTFY API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the lmktfyconfig user to use
      --username="": Username for basic authentication to the API server.
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging
```

### SEE ALSO
* [lmktfyctl](lmktfyctl.md)

