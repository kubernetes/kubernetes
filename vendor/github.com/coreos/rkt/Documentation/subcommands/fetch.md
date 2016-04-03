# rkt fetch

rkt uses HTTPS to locate and download remote ACIs and their attached signatures.
If the ACI exists locally, it won't be re-downloaded.

## Fetch with Meta Discovery

The easiest way to fetch an ACI is through meta discovery.
rkt will find and download the ACI and signature from a location that the creator has published on their website.
The [ACI discovery mechanism is detailed in the App Container specification][appc-discovery].

If you have previously trusted the image creator, it will be downloaded and verified:

```
# rkt fetch coreos.com/etcd:v2.0.0
rkt: searching for app image coreos.com/etcd:v2.0.0
rkt: fetching image from https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.aci
Downloading aci: [=======================================      ] 3.25 MB/3.7 MB
Downloading signature from https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.sig
rkt: signature verified:
  CoreOS ACI Builder <release@coreos.com>
sha512-fa1cb92dc276b0f9bedf87981e61ecde
```

If you haven't trusted the creator, it will be downloaded but not verified:

```
# rkt fetch coreos.com/etcd:v2.0.0
rkt: searching for app image coreos.com/etcd:v2.0.0
rkt: fetching image from https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.aci
Downloading aci: [=======================================      ] 3.25 MB/3.7 MB
Downloading signature from https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.sig
rkt: fetching image from https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.aci
sha512-fa1cb92dc276b0f9bedf87981e61ecde
```

## Fetch from Specific Location

If you already know where an image is stored, you can fetch it directly:

```
# rkt fetch https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.aci
rkt: fetching image from https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.aci
Downloading aci: [=======================================      ] 3.25 MB/3.7 MB
Downloading signature from https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.sig
rkt: fetching image from https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.aci
sha512-fa1cb92dc276b0f9bedf87981e61ecde
```

## Fetch from a Docker registry

If you want to run an existing Docker image, you can fetch from a Docker registry.
rkt will download and convert the image to ACI.

```
# rkt --insecure-options=image fetch docker://busybox
rkt: fetching image from docker://busybox
rkt: warning: image signature verification has been disabled
Downloading layer: 4986bf8c15363d1c5d15512d5266f8777bfba4974ac56e3270e7760f6f0a8125
Downloading layer: ea13149945cb6b1e746bf28032f02e9b5a793523481a0a18645fc77ad53c4ea2
Downloading layer: df7546f9f060a2268024c8a230d8639878585defcc1bc6f79d2728a13957871b
Downloading layer: 511136ea3c5a64f264b78b5433614aec563103b4d4702f3ba7d4d2698e22c158
sha512-c4010045aec65aefa74770ef2bb648d9
```

Docker images do not support signature verification.

## Image fetching behavior

When fetching, rkt will try to avoid unnecessary network transfers.
For example, if an image is already in the local store, rkt will use HTTP's ETag and Cache-Control to avoid downloading it again unless the image was updated on the remote server.

This behavior can be changed by using the `--store-only` and `--no-store` flags.
Their meanings are detailed in the [image fetching behavior](../image-fetching-behavior.md) documentation.

## Authentication

If you want to download an image from a private repository, then you will often need to pass credentials to be able to access it.
rkt currently supports authentication for fetching images via https:// or docker:// protocols.
To specify credentials you will have to write some configuration files.
You can find the format of the configuration file and examples in the [configuration documentation](../configuration.md).
Note that the configuration kind for images downloaded via https:// and images downloaded via docker:// is different.

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--full` |  `false` | `true` or `false` | Print the full image hash after fetching |
| `--no-store` |  `false` | `true` or `false` | Fetch images ignoring the local store. See [image fetching behavior](../image-fetching-behavior.md) |
| `--signature` |  `` | A file path | Local signature file to use in validating the preceding image |
| `--store-only` |  `false` | `true` or `false` | Use only available images in the store (do not discover or download from remote URLs). See [image fetching behavior](../image-fetching-behavior.md) |

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

[appc-discovery]: https://github.com/appc/spec/blob/master/spec/discovery.md
