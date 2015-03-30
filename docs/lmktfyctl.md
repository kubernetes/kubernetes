## lmktfyctl

lmktfyctl controls the LMKTFY cluster manager

### Synopsis


lmktfyctl controls the LMKTFY cluster manager.

Find more information at https://github.com/GoogleCloudPlatform/lmktfy.

```
lmktfyctl
```

### Options

```
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the lmktfyconfig cluster to use
      --context="": The name of the lmktfyconfig context to use
  -h, --help=false: help for lmktfyctl
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
* [lmktfyctl-get](lmktfyctl-get.md)
* [lmktfyctl-describe](lmktfyctl-describe.md)
* [lmktfyctl-create](lmktfyctl-create.md)
* [lmktfyctl-update](lmktfyctl-update.md)
* [lmktfyctl-delete](lmktfyctl-delete.md)
* [lmktfyctl-namespace](lmktfyctl-namespace.md)
* [lmktfyctl-log](lmktfyctl-log.md)
* [lmktfyctl-rollingupdate](lmktfyctl-rollingupdate.md)
* [lmktfyctl-resize](lmktfyctl-resize.md)
* [lmktfyctl-exec](lmktfyctl-exec.md)
* [lmktfyctl-port-forward](lmktfyctl-port-forward.md)
* [lmktfyctl-proxy](lmktfyctl-proxy.md)
* [lmktfyctl-run-container](lmktfyctl-run-container.md)
* [lmktfyctl-stop](lmktfyctl-stop.md)
* [lmktfyctl-expose](lmktfyctl-expose.md)
* [lmktfyctl-label](lmktfyctl-label.md)
* [lmktfyctl-config](lmktfyctl-config.md)
* [lmktfyctl-clusterinfo](lmktfyctl-clusterinfo.md)
* [lmktfyctl-apiversions](lmktfyctl-apiversions.md)
* [lmktfyctl-version](lmktfyctl-version.md)

