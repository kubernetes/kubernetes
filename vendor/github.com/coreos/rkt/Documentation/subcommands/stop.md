# rkt stop

Given a list of pod UUIDs, rkt stop will shut them down, for the shipped stage1 images, this means:

* default systemd-nspawn stage1: the apps in the pod receive a TERM signal and, after a timeout, a KILL signal.
* kvm stage1: the virtual machine is shut down with `systemctl halt`.
* rkt fly stage1: the app receives a TERM signal.

The `--force` flag will stop a pod forcibly, that is:

* default systemd-nspawn stage1: the container is killed.
* kvm stage1: the qemu or lkvm process receives a KILL signal.
* rkt fly stage1: the app receives a KILL signal.

```
# rkt stop 387fc8eb cbbf5c01
"387fc8eb-eabd-4e77-b080-d8c0001eb50c"
"cbbf5c01-dd52-4ccc-a1e0-cfd8f1e88418"
# rkt stop --force 93e516b0
"93e516b0-e84b-40cf-a45b-531b14dfcce2"
```

The `--uuid-file` flag may be used to pass a text file with UUID to `stop` command.
This can be paired with `--uuid-file-save` flag to stop pods by name:

```
rkt run --uuid-file-save=/run/rkt-uuids/mypod ...
rkt stop --uuid-file=/run/rkt-uuids/mypod
```

## Other ways to stop a rkt pod

If you started rkt as a systemd service, you can stop the pod with `systemctl stop`.

If you started rkt interactively:

* For a stage1 with systemd-nspawn, you can stop the pod by pressing `^]` three times within 5 seconds.
If you're using systemd on the host, you can also use `machinectl` with the `poweroff` or `terminate` subcommand.
* For a stage1 with kvm, you can stop the pod by pressing Ctrl+A and then x.

