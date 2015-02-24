## kubectl

kubectl controls the Kubernetes cluster manager

### Synopsis

```
kubectl controls the Kubernetes cluster manager.

Find more information at https://github.com/GoogleCloudPlatform/kubernetes.
```

kubectl

### Options

```
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the kubeconfig cluster to use
      --context="": The name of the kubeconfig context to use
  -h, --help=false: help for kubectl
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --kubeconfig="": Path to the kubeconfig file to use for CLI requests.
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --password="": Password for basic authentication to the API server.
  -s, --server="": The address and port of the Kubernetes API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the kubeconfig user to use
      --username="": Username for basic authentication to the API server.
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging
```

### SEE ALSO
* [kubectl-version](kubectl-version.md)
* [kubectl-proxy](kubectl-proxy.md)
* [kubectl-get](kubectl-get.md)
* [kubectl-describe](kubectl-describe.md)
* [kubectl-create](kubectl-create.md)
* [kubectl-update](kubectl-update.md)
* [kubectl-delete](kubectl-delete.md)
* [kubectl-config](kubectl-config.md)
* [kubectl-namespace](kubectl-namespace.md)
* [kubectl-log](kubectl-log.md)
* [kubectl-rollingupdate](kubectl-rollingupdate.md)
* [kubectl-resize](kubectl-resize.md)
* [kubectl-exec](kubectl-exec.md)
* [kubectl-port-forward](kubectl-port-forward.md)
* [kubectl-run-container](kubectl-run-container.md)
* [kubectl-stop](kubectl-stop.md)
* [kubectl-expose](kubectl-expose.md)
* [kubectl-label](kubectl-label.md)

