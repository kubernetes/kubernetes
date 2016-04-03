# rkt prepare

rkt can prepare images to run in a pod.
This means it will fetch (if necessary) the images, extract them in its internal tree store, and allocate a pod UUID.
If overlay fs is not supported or disabled, it will also copy the tree in the pod rootfs.

In this way, the pod is ready to be launched immediately by the [run-prepared](run-prepared.md) command.

Running `rkt prepare` + `rkt run-prepared` is semantically equivalent to running [rkt run](run.md).
Therefore, the supported arguments are mostly the same as in `run` except runtime arguments like `--interactive` or `--mds-register`.

## Example

```
# rkt prepare coreos.com/etcd:v2.0.10
rkt prepare coreos.com/etcd:v2.0.10
rkt: using image from local store for image name coreos.com/rkt/stage1-coreos:1.2.1
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
| `--exec` |  `` | A path | Override the exec command for the preceding image |
| `--inherit-env` |  `false` | `true` or `false` | Inherit all environment variables not set by apps |
| `--mount` |  `` | Mount syntax (`volume=NAME,target=PATH`). See [Mounting Volumes without Mount Points](run.md#mounting-volumes-without-mount-points) | Mount point binding a volume to a path within an app |
| `--no-overlay` |  `false` | `true` or `false` | Disable overlay filesystem |
| `--no-store` |  `false` | `true` or `false` | Fetch images, ignoring the local store. See [image fetching behavior](../image-fetching-behavior.md) |
| `--pod-manifest` |  `` | A path | The path to the pod manifest. If it's non-empty, then only `--net`, `--no-overlay` and `--interactive` will have effect |
| `--port` |  `` | A port number | Ports to expose on the host (requires [contained network](https://github.com/coreos/rkt/blob/master/Documentation/networking.md#contained-mode)). Syntax: --port=NAME:HOSTPORT |
| `--private-users` |  `false` | `true` or `false` | Run within user namespaces (experimental) |
| `--quiet` |  `false` | `true` or `false` | Suppress superfluous output on stdout, print only the UUID on success |
| `--set-env` |  `` | An environment variable. Syntax `NAME=VALUE` | An environment variable to set for apps |
| `--signature` |  `` | A file path | Local signature file to use in validating the preceding image |
| `--stage1-url` |  `` | A URL to a stage1 image. HTTP/HTTPS/File/Docker URLs are supported | Image to use as stage1 |
| `--stage1-path` |  `` | A path to a stage1 image. Absolute and relative paths are supported | Image to use as stage1 |
| `--stage1-name` |  `` | A name of a stage1 image. Will perform a discovery if the image is not in the store | Image to use as stage1 |
| `--stage1-hash` |  `` | A hash of a stage1 image. The image must exist in the store | Image to use as stage1 |
| `--stage1-from-dir` |  `` | A stage1 image file inside the default stage1 images directory | Image to use as stage1 |
| `--store-only` |  `false` | `true` or `false` | Use only available images in the store (do not discover or download from remote URLs). See [image fetching behavior](../image-fetching-behavior.md) |
| `--volume` |  `` | Volume syntax (`NAME,kind=KIND,source=PATH,readOnly=BOOL`). See [Mount Volumes into a Pod](run.md#mount-volumes-into-a-pod) | Volumes to make available in the pod |

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
