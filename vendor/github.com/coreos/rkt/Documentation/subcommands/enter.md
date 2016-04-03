# rkt enter

Given a pod UUID, if you want to enter a running pod to explore its filesystem or see what's running you can use rkt enter.

```
# rkt enter 76dc6286
Pod contains multiple apps:
        redis
        etcd
Unable to determine app name: specify app using "rkt enter --app= ..."

# rkt enter --app=redis 76dc6286
No command specified, assuming "/bin/bash"
root@rkt-76dc6286-f672-45f2-908c-c36dcd663560:/# ls
bin   data  entrypoint.sh  home  lib64  mnt  proc  run   selinux  sys  usr
boot  dev   etc            lib   media  opt  root  sbin  srv      tmp  var
```

## Use a Custom Stage 1

rkt is designed and intended to be modular, using a [staged architecture](../devel/architecture.md).

You can use a custom stage1 by using the `--stage1-{url,path,name,hash,from-dir}` flags.

```
# rkt --stage1-path=/tmp/stage1.aci run coreos.com/etcd:v2.0.0
```

For more details see the [hacking documentation](../hacking.md).

## Run a Pod in the Background

Work in progress. Please contribute!

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--app` |  `` | Name of an app | Name of the app to enter within the specified pod |

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