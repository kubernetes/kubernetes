# rkt rm

Cleans up all resources (files, network objects) associated with a pod just like `rkt gc`.
This command can be used to immediately free resources without waiting for garbage collection to run.

```
rkt rm c138310f
```

Instead of passing UUID on command line, rm command can read the UUID from a text file.
This can be paired with `--uuid-file-save` to remove pods by name:

```
rkt run --uuid-files-save=/run/rkt-uuids/mypod ...
rkt rm --uuid-file=/run/rkt-uuids/mypod
```

### Global options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--debug` |  `false` | `true` or `false` | Prints out more debug information to `stderr` |
| `--dir` | `/var/lib/rkt` | A directory path | Path to the `rkt` data directory |
| `--insecure-options` |  none | **none**: All security features are enabled<br/>**http**: Allow HTTP connections. Be warned that this will send any credentials as clear text.<br/>**image**: Disables verifying image signatures<br/>**tls**: Accept any certificate from the server and any host name in that certificate<br/>**ondisk**: Disables verifying the integrity of the on-disk, rendered image before running. This significantly speeds up start time.<br/>**all**: Disables all security checks | Comma-separated list of security features to disable |
| `--local-config` |  `/etc/rkt` | A directory path | Path to the local configuration directory |
| `--system-config` |  `/usr/lib/rkt` | A directory path | Path to the system configuration directory |
| `--trust-keys-from-https` |  `false` | `true` or `false` | Automatically trust gpg keys fetched from https |
| `--user-config` |  `` | A directory path | Path to the user configuration directory |