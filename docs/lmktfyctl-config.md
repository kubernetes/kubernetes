## lmktfyctl config

config modifies .lmktfyconfig files

### Synopsis


config modifies .lmktfyconfig files using subcommands like "lmktfyctl config set current-context my-context"

```
lmktfyctl config SUBCOMMAND
```

### Options

```
      --envvar=false: use the .lmktfyconfig from $LMKTFYCONFIG
      --global=false: use the .lmktfyconfig from /home/username
  -h, --help=false: help for config
      --lmktfyconfig="": use a particular .lmktfyconfig file
      --local=false: use the .lmktfyconfig in the current directory
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
* [lmktfyctl-config-view](lmktfyctl-config-view.md)
* [lmktfyctl-config-set-cluster](lmktfyctl-config-set-cluster.md)
* [lmktfyctl-config-set-credentials](lmktfyctl-config-set-credentials.md)
* [lmktfyctl-config-set-context](lmktfyctl-config-set-context.md)
* [lmktfyctl-config-set](lmktfyctl-config-set.md)
* [lmktfyctl-config-unset](lmktfyctl-config-unset.md)
* [lmktfyctl-config-use-context](lmktfyctl-config-use-context.md)

