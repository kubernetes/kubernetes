# rkt list

You can list all rkt pods.

```
$ rkt list
UUID        APP     IMAGE NAME               STATE      CREATED        STARTED         NETWORKS
5bc080ca    redis   redis                    running    2 minutes ago  41 seconds ago  default:ip4=172.16.28.7
            etcd    coreos.com/etcd:v2.0.9
3089337c    nginx   nginx                    exited     9 minutes ago  2 minutes ago
```

You can view the full UUID as well as the image's ID by using the `--full` flag.

```
$ rkt list --full
UUID                                   APP     IMAGE NAME              IMAGE ID              STATE      CREATED                             STARTED                             NETWORKS
5bc080cav-9e03-480d-b705-5928af396cc5  redis   redis                   sha512-91e98d7f1679   running    2016-01-25 17:42:32.563 +0100 CET   2016-01-25 17:44:05.294 +0100 CET   default:ip4=172.16.28.7
                                       etcd    coreos.com/etcd:v2.0.9  sha512-a03f6bad952b
3089337c4-8021-119b-5ea0-879a7c694de4  nginx   nginx                   sha512-32ad6892f21a   exited     2016-01-25 17:36:40.203 +0100 CET   2016-01-25 17:42:15.1 +0100 CET
```

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--full` |  `false` | `true` or `false` | Use long output format |
| `--no-legend` |  `false` | `true` or `false` | Suppress a legend with the list |

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
