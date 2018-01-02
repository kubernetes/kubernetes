# rkt image

## rkt image cat-manifest

For debugging or inspection you may want to extract an ACI manifest to stdout.

```
# rkt image cat-manifest coreos.com/etcd
{
  "acVersion": "0.8.10",
  "acKind": "ImageManifest",
...
```

### Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--pretty-print` |  `true` | `true` or `false` | Apply indent to format the output |

## rkt image export

There are cases where you might want to export the ACI from the store to copy to another machine, file server, etc.

```
# rkt image export coreos.com/etcd etcd.aci
$ tar xvf etcd.aci
```

NOTES:

- A matching image must be fetched before doing this operation, rkt will not attempt to download an image first, this subcommand will incur no-network I/O.
- The exported ACI file might be different than the original one because rkt image export always returns uncompressed ACIs.

### Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--overwrite` |  `false` | `true` or `false` | Overwrite output ACI |

## rkt image extract/render

For debugging or inspection you may want to extract an ACI to a directory on disk.
There are a few different options depending on your use case but the basic command looks like this:

```
# rkt image extract coreos.com/etcd etcd-extracted
# find etcd-extracted
etcd-extracted
etcd-extracted/manifest
etcd-extracted/rootfs
etcd-extracted/rootfs/etcd
etcd-extracted/rootfs/etcdctl
...
```

NOTE: Like with rkt image export, a matching image must be fetched before doing this operation.

Now there are some flags that can be added to this:

To get just the rootfs use:

```
# rkt image extract --rootfs-only coreos.com/etcd etcd-extracted
# find etcd-extracted
etcd-extracted
etcd-extracted/etcd
etcd-extracted/etcdctl
...
```

If you want the image rendered as it would look ready-to-run inside of the rkt stage2 then use `rkt image render`.
NOTE: this will not use overlayfs or any other mechanism.
This is to simplify the cleanup: to remove the extracted files you can run a simple `rm -Rf`.

### Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--overwrite` |  `false` | `true` or `false` | Overwrite output directory |
| `--rootfs-only` |  `false` | `true` or `false` | Extract rootfs only |

## rkt image gc

You can garbage collect the rkt store to clean up unused internal data and remove old images.

By default, images not used in the last 24h will be removed.
This can be configured with the `--grace-period` flag.

```
# rkt image gc --grace-period 48h
rkt: removed treestore "deps-sha512-219204dd54481154aec8f6eafc0f2064d973c8a2c0537eab827b7414f0a36248"
rkt: removed treestore "deps-sha512-3f2a1ad0e9739d977278f0019b6d7d9024a10a2b1166f6c9fdc98f77a357856d"
rkt: successfully removed aci for image: "sha512-e39d4089a224718c41e6bef4c1ac692a6c1832c8c69cf28123e1f205a9355444" ("coreos.com/rkt/stage1")
rkt: successfully removed aci for image: "sha512-0648aa44a37a8200147d41d1a9eff0757d0ac113a22411f27e4e03cbd1e84d0d" ("coreos.com/etcd")
rkt: 2 image(s) successfully removed
```

### Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--grace-period` |  `24h0m0s` | A time | Duration to wait since an image was last used before removing it |

## rkt image list

You can get a list of images in the local store with their keys, names and import times.

```
# rkt image list
ID                       NAME                            IMPORT TIME     LAST USED      SIZE    LATEST
sha512-91e98d7f1679      coreos.com/etcd:v2.0.9          6 days ago      2 minutes ago  12MiB   false
sha512-a03f6bad952b      coreos.com/rkt/stage1:0.7.0     55 minutes ago  2 minutes ago  143MiB  false
```

A more detailed output can be had by adding the `--full` flag:

```
ID                                                                        NAME                                         IMPORT TIME                          LAST USED                           SIZE       LATEST
sha512-96323da393621d846c632e71551b77089ac0b004ceb5c2362be4f5ced2212db9   registry-1.docker.io/library/redis:latest    2015-12-14 12:30:33.652 +0100 CET    2015-12-14 12:33:40.812 +0100 CET   113309184  true
```

### Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--fields` |  `id,name,importtime,lastused,size,latest` | A comma-separated list with one or more of `id`, `name`, `importtime`, `lastused`, `size`, `latest` | Comma-separated list of fields to display |
| `--full` |  `false` | `true` or `false` | Use long output format |
| `--no-legend` |  `false` | `true` or `false` | Suppress a legend with the list |
| `--order` |  `asc` | `asc` or `desc` | Choose the sorting order if at least one sort field is provided (`--sort`) |
| `--sort` |  `importtime` | A comma-separated list with one or more of `id`, `name`, `importtime`, `lastused`, `size`, `latest` | Sort the output according to the provided comma-separated list of fields |

## rkt image rm

Given multiple image IDs or image names you can remove them from the local store.

```
# rkt image rm sha512-a03f6bad952b coreos.com/etcd
rkt: successfully removed aci for image: "sha512-a03f6bad952bd548c2a57a5d2fbb46679aff697ccdacd6c62e1e1068d848a9d4" ("coreos.com/rkt/stage1")
rkt: successfully removed aci for image: "sha512-91e98d7f167905b69cce91b163963ccd6a8e1c4bd34eeb44415f0462e4647e27" ("coreos.com/etcd")
rkt: 2 image(s) successfully removed
```

## Global options

See the table with [global options in general commands documentation][global-options].


[global-options]: ../commands.md#global-options
