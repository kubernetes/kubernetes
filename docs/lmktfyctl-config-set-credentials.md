## lmktfyctl config set-credentials

Sets a user entry in .lmktfyconfig

### Synopsis


Sets a user entry in .lmktfyconfig
Specifying a name that already exists will merge new fields on top of existing values.

  Client-certificate flags:
    --client-certificate=certfile --client-key=keyfile

  Bearer token flags:
    --token=bearer_token

  Basic auth flags:
    --username=basic_user --password=basic_password

  Bearer token and basic auth are mutually exclusive.


```
lmktfyctl config set-credentials NAME [--auth-path=/path/to/authfile] [--client-certificate=path/to/certfile] [--client-key=path/to/keyfile] [--token=bearer_token] [--username=basic_user] [--password=basic_password]
```

### Examples

```
// Set only the "client-key" field on the "cluster-admin"
// entry, without touching other values:
$ lmktfyctl set-credentials cluster-admin --client-key=~/.lmktfy/admin.key

// Set basic auth for the "cluster-admin" entry
$ lmktfyctl set-credentials cluster-admin --username=admin --password=uXFGweU9l35qcif

// Embed client certificate data in the "cluster-admin" entry
$ lmktfyctl set-credentials cluster-admin --client-certificate=~/.lmktfy/admin.crt --embed-certs=true
```

### Options

```
      --auth-path=: auth-path for the user entry in .lmktfyconfig
      --client-certificate=: path to client-certificate for the user entry in .lmktfyconfig
      --client-key=: path to client-key for the user entry in .lmktfyconfig
      --embed-certs=false: embed client cert/key for the user entry in .lmktfyconfig
  -h, --help=false: help for set-credentials
      --password=: password for the user entry in .lmktfyconfig
      --token=: token for the user entry in .lmktfyconfig
      --username=: username for the user entry in .lmktfyconfig
```

### Options inherrited from parent commands

```
      --alsologtostderr=false: log to standard error as well as files
      --api-version="": The API version to use when talking to the server
      --certificate-authority="": Path to a cert. file for the certificate authority.
      --cluster="": The name of the lmktfyconfig cluster to use
      --context="": The name of the lmktfyconfig context to use
      --envvar=false: use the .lmktfyconfig from $LMKTFYCONFIG
      --global=false: use the .lmktfyconfig from /home/username
      --insecure-skip-tls-verify=false: If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.
      --lmktfyconfig="": use a particular .lmktfyconfig file
      --local=false: use the .lmktfyconfig in the current directory
      --log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
      --log_dir=: If non-empty, write log files in this directory
      --log_flush_frequency=5s: Maximum number of seconds between log flushes
      --logtostderr=true: log to standard error instead of files
      --match-server-version=false: Require server version to match client version
      --namespace="": If present, the namespace scope for this CLI request.
  -s, --server="": The address and port of the LMKTFY API server
      --stderrthreshold=2: logs at or above this threshold go to stderr
      --user="": The name of the lmktfyconfig user to use
      --v=0: log level for V logs
      --validate=false: If true, use a schema to validate the input before sending it
      --vmodule=: comma-separated list of pattern=N settings for file-filtered logging
```

### SEE ALSO
* [lmktfyctl-config](lmktfyctl-config.md)

