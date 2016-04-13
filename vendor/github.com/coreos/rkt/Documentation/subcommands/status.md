# rkt status

Given a pod UUID, you can get the exit status of its apps.
Note that the apps are prefixed by `app-`.

```
$ rkt status 66ceb509
state=exited
created=2016-01-26 14:23:34.631 +0100 CET
started=2016-01-26 14:23:34.744 +0100 CET
pid=16964
exited=true
app-redis=0
app-etcd=0
```

If the pod is still running, you can wait for it to finish and then get the status with `rkt status --wait UUID`

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--wait` |  `false` | `true` or `false` | Toggle waiting for the pod to exit |

## Global options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--debug` |  `false` | `true` or `false` | Prints out more debug information to `stderr` |
| `--dir` | `/var/lib/rkt` | A directory path | Path to the `rkt` data directory |
| `--insecure-options` |  none | **none**: All security features are enabled<br/>**http**: Allow HTTP connections. Be warned that this will send any credentials as clear text.<br/>**image**: Disables verifying image signatures<br/>**tls**: Accept any certificate from the server and any host name in that certificate<br/>**ondisk**: Disables verifying the integrity of the on-disk, rendered image before running. This significantly speeds up start time.<br/>**all**: Disables all security checks | Comma-separated list of security features to disable |
| `--local-config` |  `/etc/rkt` | A directory path | Path to the local configuration directory |
| `--system-config` |  `/usr/lib/rkt` | A directory path | Path to the system configuration directory |
| `--trust-keys-from-https` |  `false` | `true` or `false` | Automatically trust gpg keys fetched from https |
| `--user-config` |  `` | A directory path | Path to the user configuration directory |