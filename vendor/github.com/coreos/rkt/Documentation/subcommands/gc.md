# rkt gc

rkt has a built-in garbage collection command that is designed to be run periodically from a timer or cron job.
Stopped pods are moved to the garbage and cleaned up during a subsequent garbage collection pass.
Each `gc` pass removes any pods remaining in the garbage past the grace period.
[Read more about the pod lifecycle][gc-docs].

```
# rkt gc --grace-period=30m0s
Moving pod "21b1cb32-c156-4d26-82ae-eda1ab60f595" to garbage
Moving pod "5dd42e9c-7413-49a9-9113-c2a8327d08ab" to garbage
Moving pod "f07a4070-79a9-4db0-ae65-a090c9c393a3" to garbage
```

On the next pass, the pods are removed:

```
# rkt gc --grace-period=30m0s
Garbage collecting pod "21b1cb32-c156-4d26-82ae-eda1ab60f595"
Garbage collecting pod "5dd42e9c-7413-49a9-9113-c2a8327d08ab"
Garbage collecting pod "f07a4070-79a9-4db0-ae65-a090c9c393a3"
```

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--expire-prepared` |  `24h0m0s` | A time | Duration to wait before expiring prepared pods |
| `--grace-period` |  `30m0s` | A time | Duration to wait before discarding inactive pods from garbage |
| `--mark-only` | `false` | `true` or `false` | If set to true, only the "mark" phase of the garbage collection process will be formed (i.e., exited/aborted pods will be moved to the garbage, but nothing will be deleted) |

## Global options

See the table with [global options in general commands documentation][global-options].


[gc-docs]: ../devel/pod-lifecycle.md#garbage-collection
[global-options]: ../commands.md#global-options
