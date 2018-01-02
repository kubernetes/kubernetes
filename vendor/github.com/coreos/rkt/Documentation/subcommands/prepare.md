# rkt prepare

rkt can prepare images to run in a pod.
This means it will fetch (if necessary) the images, extract them in its internal tree store, and allocate a pod UUID.
If overlay fs is not supported or disabled, it will also copy the tree in the pod rootfs.

Support for overlay fs will be auto-detected if `--no-overlay` is set to `false`. If an unsupported filesystem is detected, rkt will print a warning message and continue preparing the pod by falling back in non-overlay mode as described above:

```
# rkt prepare --insecure-options=image docker://busybox --exec=/bin/sh
image: using image from local store for image name coreos.com/rkt/stage1-coreos:1.25.0
image: remote fetching from URL "docker://busybox"
Downloading sha256:8ddc19f1652 [===============================] 668 KB / 668 KB
prepare: disabling overlay support: "unsupported filesystem: missing d_type support"
```

The following conditions can lead to non-overlay mode:

The data directory (usually `/var/lib/rkt`) is on ...
- an AUFS filesystem
- a ZFS filesystem
- a XFS filesystem having `ftype=0`
- a file system where the `d_type` field is set to `DT_UNKNOWN`, see getdents(2)

In this way, the pod is ready to be launched immediately by the [run-prepared][run-prepared] command.

Running `rkt prepare` + `rkt run-prepared` is semantically equivalent to running [rkt run][run].
Therefore, the supported arguments are mostly the same as in `run` except runtime arguments like `--interactive` or `--mds-register`.

## Example

```
# rkt prepare coreos.com/etcd:v2.0.10
rkt prepare coreos.com/etcd:v2.0.10
rkt: using image from local store for image name coreos.com/rkt/stage1-coreos:1.25.0
rkt: searching for app image coreos.com/etcd:v2.0.10
rkt: remote fetching from url https://github.com/coreos/etcd/releases/download/v2.0.10/etcd-v2.0.10-linux-amd64.aci
prefix: "coreos.com/etcd"
key: "https://coreos.com/dist/pubkeys/aci-pubkeys.gpg"
gpg key fingerprint is: 8B86 DE38 890D DB72 9186  7B02 5210 BD88 8818 2190
	CoreOS ACI Builder <release@coreos.com>
Key "https://coreos.com/dist/pubkeys/aci-pubkeys.gpg" already in the keystore
Downloading signature from https://github.com/coreos/etcd/releases/download/v2.0.10/etcd-v2.0.10-linux-amd64.aci.asc
Downloading signature: [=======================================] 819 B/819 B
Downloading ACI: [=============================================] 3.79 MB/3.79 MB
rkt: signature verified:
  CoreOS ACI Builder <release@coreos.com>
c9fad0e6-8236-4fc2-ad17-55d0a4c7d742
```

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--user-annotation` | none | annotation add to the app's UserAnnotations field | Set the app's annotations (example: '--annotation=foo=bar'). |
| `--caps-remove` | none | capability to remove (example: '--caps-remove=CAP\_SYS\_CHROOT,CAP\_MKNOD') | Capabilities to remove from the process's capabilities bounding set, all others from the default set will be included |
| `--caps-retain` | none | capability to retain (example: '--caps-remove=CAP\_SYS\_ADMIN,CAP\_NET\_ADMIN') | Capabilities to retain in the process's capabilities bounding set, all others will be removed |
| `--environment` | none | environment variables add to the app's environment variables | Set the app's environment variables (example: '--environment=foo=bar'). |
| `--exec` | none | Path to executable | Override the exec command for the preceding image. |
| `--group` | root | gid, groupname or file path | Group override for the preceding image (example: '--group=group') |
| `--inherit-env` | `false` | `true` or `false` | Inherit all environment variables not set by apps. |
| `--user-label` | none | label add to the apps' UserLabels field | Set the app's labels (example: '--label=foo=bar'). |
| `--mount` | none | Mount syntax (ex. `--mount volume=NAME,target=PATH`) | Mount point binding a volume to a path within an app. See [Mounting Volumes without Mount Points][vol-no-mount]. |
| `--name` | none | Name of the app | Set the name of the app (example: '--name=foo'). If not set, then the app name default to the image's name |
| `--no-overlay` | `false` | `true` or `false` | Disable the overlay filesystem. |
| `--pull-policy` | `new` | `never`, `new`, or `update` | Sets the policy for when to fetch an image. See [image fetching behavior][img-fetch] |
| `--pod-manifest` | none | A path | The path to the pod manifest. If it's non-empty, then only `--net`, `--no-overlay` and `--interactive` will have effect. |
| `--port` | none | A port name and number pair | Container port name to expose through host port number. Requires [contained network][contained]. Syntax: `--port=NAME:HOSTPORT` The NAME is that given in the ACI. By convention, Docker containers' EXPOSEd ports are given a name formed from the port number, a hyphen, and the protocol, e.g., `80-tcp`, giving something like `--port=80-tcp:8080` |
| `--private-users` |  `false` | `true` or `false` | Run within user namespaces |
| `--quiet` |  `false` | `true` or `false` | Suppress superfluous output on stdout, print only the UUID on success |
| `--set-env` |  `` | An environment variable. Syntax `NAME=VALUE` | An environment variable to set for apps |
| `--set-env-file` |  `` | Path of an environment variables file | Environment variables to set for apps |
| `--signature` |  `` | A file path | Local signature file to use in validating the preceding image |
| `--stage1-url` |  `` | A URL to a stage1 image. HTTP/HTTPS/File/Docker URLs are supported | Image to use as stage1 |
| `--stage1-path` |  `` | A path to a stage1 image. Absolute and relative paths are supported | Image to use as stage1 |
| `--stage1-name` |  `` | A name of a stage1 image. Will perform a discovery if the image is not in the store | Image to use as stage1 |
| `--stage1-hash` |  `` | A hash of a stage1 image. The image must exist in the store | Image to use as stage1 |
| `--stage1-from-dir` |  `` | A stage1 image file inside the default stage1 images directory | Image to use as stage1 |
| `--user` | none | uid, username or file path | user override for the preceding image (example: '--user=user') |
| `--volume` |  `` | Volume syntax (`NAME,kind=KIND,source=PATH,readOnly=BOOL,recursive=BOOL`). See [Mount Volumes into a Pod][mount-vol] | Volumes to make available in the pod |

## Global options

See the table with [global options in general commands documentation][global-options].


[contained]: ../networking/overview.md#contained-mode
[global-options]: ../commands.md#global-options
[img-fetch]: ../image-fetching-behavior.md
[mount-vol]: run.md#mount-volumes-into-a-pod
[run]: run.md
[run-prepared]: run-prepared.md
[vol-no-mount]: run.md#mounting-volumes-without-mount-points
