# Snapshot Migration

You can migrate a snapshot of your data from a v0.4.9+ cluster into a new etcd 2.2 cluster using a snapshot migration. After snapshot migration, the etcd indexes of your data will change. Many etcd applications rely on these indexes to behave correctly. This operation should only be done while all etcd applications are stopped.

To get started get the newest data snapshot from the 0.4.9+ cluster:

```
curl http://cluster.example.com:4001/v2/migration/snapshot > backup.snap
```

Now, import the snapshot into your new cluster:

```
etcdctl --endpoint new_cluster.example.com import --snap backup.snap
```

If you have a large amount of data, you can specify more concurrent works to copy data in parallel by using `-c` flag.
If you have hidden keys to copy, you can use `--hidden` flag to specify. For example fleet uses `/_coreos.com/fleet` so to import those keys use `--hidden /_coreos.com`.

And the data will quickly copy into the new cluster:

```
entering dir: /
entering dir: /foo
entering dir: /foo/bar
copying key: /foo/bar/1 1
entering dir: /
entering dir: /foo2
entering dir: /foo2/bar2
copying key: /foo2/bar2/2 2
```
