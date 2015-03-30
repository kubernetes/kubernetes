## lmktfyctl config set-cluster

Sets a cluster entry in .lmktfyconfig

### Synopsis


Sets a cluster entry in .lmktfyconfig.
Specifying a name that already exists will merge new fields on top of existing values for those fields.

```
lmktfyctl config set-cluster NAME [--server=server] [--certificate-authority=path/to/certficate/authority] [--api-version=apiversion] [--insecure-skip-tls-verify=true]
```

### Examples

```
// Set only the server field on the e2e cluster entry without touching other values.
$ lmktfyctl config set-cluster e2e --server=https://1.2.3.4

// Embed certificate authority data for the e2e cluster entry
$ lmktfyctl config set-cluster e2e --certificate-authority=~/.lmktfy/e2e/lmktfy.ca.crt

// Disable cert checking for the dev cluster entry
$ lmktfyctl config set-cluster e2e --insecure-skip-tls-verify=true
```

### Options

```
      --api-version=: api-version for the cluster entry in .lmktfyconfig
      --certificate-authority=: path to certificate-authority for the cluster entry in .lmktfyconfig
      --embed-certs=false: embed-certs for the cluster entry in .lmktfyconfig
  -h, --help=false: help for set-cluster
      --insecure-skip-tls-verify=false: insecure-skip-tls-verify for the cluster entry in .lmktfyconfig
      --server=: server for the cluster entry in .lmktfyconfig
```

### Options inherrited from parent commands

```
      --alsologtostderr=false: log to standard error as well as files
  -a, --auth-path="": Path to the auth info file. If missing, prompt the user. Only used if using https.
      --client-certificate="": Path to a client key file for TLS.
      --client-key="": Path to a client key file for TLS.
      --cluster="": The name of the lmktfyconfig cluster to use
      --context="": The name of the lmktfyconfig context to use
      --envvar=false: use the .lmktfyconfig from $LMKTFYCONFIG
      --global=false: use the .lmktfyconfig from /home/username
      --lmktfyconfig="": use a particular .lmktfyconfig file
      --local=false: use the .lmktfyconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
      --password="": Password for basic authentication to the API server.
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --token="": Bearer token for authentication to the API server.
      --user="": The name of the lmktfyconfig user to use
      --username="": Username for basic authentication to the API server.
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging
```

### SEE ALSO
* [lmktfyctl-config](lmktfyctl-config.md)

