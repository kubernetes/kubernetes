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

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--app` |  `` | Name of an app | Name of the app to enter within the specified pod |

## Global options

See the table with [global options in general commands documentation][global-options].


[global-options]: ../commands.md#global-options
