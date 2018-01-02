# rkt rm

Cleans up all resources (files, network objects) associated with a pod just like `rkt gc`.
This command can be used to immediately free resources without waiting for garbage collection to run.

```
rkt rm c138310f
```

Instead of passing UUID on command line, rm command can read the UUID from a text file.
This can be paired with `--uuid-file-save` to remove pods by name:

```
rkt run --uuid-file-save=/run/rkt-uuids/mypod ...
rkt rm --uuid-file=/run/rkt-uuids/mypod
```

### Global options

See the table with [global options in general commands documentation][global-options].


[global-options]: ../commands.md#global-options
