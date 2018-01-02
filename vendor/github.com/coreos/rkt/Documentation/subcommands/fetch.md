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

This behavior can be changed by using the `--pull-policy` flag.
Usage of this flag is detailed in the [image fetching behavior][img-fetch] documentation.

## Authentication

If you want to download an image from a private repository, then you will often need to pass credentials to be able to access it.
rkt currently supports authentication for fetching images via https:// or docker:// protocols.
To specify credentials you will have to write some configuration files.
You can find the format of the configuration file and examples in the [configuration documentation][configuration].
Note that the configuration kind for images downloaded via https:// and images downloaded via docker:// is different.

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--full` |  `false` | `true` or `false` | Print the full image hash after fetching |
| `--signature` |  `` | A file path | Local signature file to use in validating the preceding image |
| `--pull-policy` | `new` | `never`, `new`, or `update` | Sets the policy for when to fetch an image. See [image fetching behavior][img-fetch] |

## Global options

See the table with [global options in general commands documentation][global-options].


[appc-discovery]: https://github.com/appc/spec/blob/master/spec/discovery.md
[configuration]: ../configuration.md
[global-options]: ../commands.md#global-options
[img-fetch]: ../image-fetching-behavior.md
