# `tarball`

[![GoDoc](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/tarball?status.svg)](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/tarball)

This package produces tarballs that can consumed via `docker load`. Note
that this is a _different_ format from the [`legacy`](/pkg/legacy/tarball)
tarballs that are produced by `docker save`, but this package is still able to
read the legacy tarballs produced by `docker save`.

## Usage

```go
package main

import (
	"os"

	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/tarball"
)

func main() {
	// Read a tarball from os.Args[1] that contains ubuntu.
	tag, err := name.NewTag("ubuntu")
	if err != nil {
		panic(err)
	}
	img, err := tarball.ImageFromPath(os.Args[1], &tag)
	if err != nil {
		panic(err)
	}

	// Write that tarball to os.Args[2] with a different tag.
	newTag, err := name.NewTag("ubuntu:newest")
	if err != nil {
		panic(err)
	}
	f, err := os.Create(os.Args[2])
	if err != nil {
		panic(err)
	}
	defer f.Close()

	if err := tarball.Write(newTag, img, f); err != nil {
		panic(err)
	}
}
```

## Structure

<p align="center">
  <img src="/images/tarball.dot.svg" />
</p>

Let's look at what happens when we write out a tarball:


### `ubuntu:latest`

```
$ crane pull ubuntu ubuntu.tar && mkdir ubuntu && tar xf ubuntu.tar -C ubuntu && rm ubuntu.tar
$ tree ubuntu/
ubuntu/
├── 423ae2b273f4c17ceee9e8482fa8d071d90c7d052ae208e1fe4963fceb3d6954.tar.gz
├── b6b53be908de2c0c78070fff0a9f04835211b3156c4e73785747af365e71a0d7.tar.gz
├── de83a2304fa1f7c4a13708a0d15b9704f5945c2be5cbb2b3ed9b2ccb718d0b3d.tar.gz
├── f9a83bce3af0648efaa60b9bb28225b09136d2d35d0bed25ac764297076dec1b.tar.gz
├── manifest.json
└── sha256:72300a873c2ca11c70d0c8642177ce76ff69ae04d61a5813ef58d40ff66e3e7c

0 directories, 6 files
```

There are a couple interesting files here.

`manifest.json` is the entrypoint: a list of [`tarball.Descriptor`s](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/tarball#Descriptor)
that describe the images contained in this tarball.

For each image, this has the `RepoTags` (how it was pulled), a `Config` file
that points to the image's config file, a list of `Layers`, and (optionally)
`LayerSources`.

```
$ jq < ubuntu/manifest.json
[
  {
    "Config": "sha256:72300a873c2ca11c70d0c8642177ce76ff69ae04d61a5813ef58d40ff66e3e7c",
    "RepoTags": [
      "ubuntu"
    ],
    "Layers": [
      "423ae2b273f4c17ceee9e8482fa8d071d90c7d052ae208e1fe4963fceb3d6954.tar.gz",
      "de83a2304fa1f7c4a13708a0d15b9704f5945c2be5cbb2b3ed9b2ccb718d0b3d.tar.gz",
      "f9a83bce3af0648efaa60b9bb28225b09136d2d35d0bed25ac764297076dec1b.tar.gz",
      "b6b53be908de2c0c78070fff0a9f04835211b3156c4e73785747af365e71a0d7.tar.gz"
    ]
  }
]
```

The config file and layers are exactly what you would expect, and match the
registry representations of the same artifacts. You'll notice that the
`manifest.json` contains similar information as the registry manifest, but isn't
quite the same:

```
$ crane manifest ubuntu@sha256:0925d086715714114c1988f7c947db94064fd385e171a63c07730f1fa014e6f9
{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 3408,
      "digest": "sha256:72300a873c2ca11c70d0c8642177ce76ff69ae04d61a5813ef58d40ff66e3e7c"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 26692096,
         "digest": "sha256:423ae2b273f4c17ceee9e8482fa8d071d90c7d052ae208e1fe4963fceb3d6954"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 35365,
         "digest": "sha256:de83a2304fa1f7c4a13708a0d15b9704f5945c2be5cbb2b3ed9b2ccb718d0b3d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 852,
         "digest": "sha256:f9a83bce3af0648efaa60b9bb28225b09136d2d35d0bed25ac764297076dec1b"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 163,
         "digest": "sha256:b6b53be908de2c0c78070fff0a9f04835211b3156c4e73785747af365e71a0d7"
      }
   ]
}
```

This makes it difficult to maintain image digests when roundtripping images
through the tarball format, so it's not a great format if you care about
provenance.

The ubuntu example didn't have any `LayerSources` -- let's look at another image
that does.

### `hello-world:nanoserver`

```
$ crane pull hello-world:nanoserver@sha256:63c287625c2b0b72900e562de73c0e381472a83b1b39217aef3856cd398eca0b nanoserver.tar
$ mkdir nanoserver && tar xf nanoserver.tar -C nanoserver && rm nanoserver.tar
$ tree nanoserver/
nanoserver/
├── 10d1439be4eb8819987ec2e9c140d44d74d6b42a823d57fe1953bd99948e1bc0.tar.gz
├── a35da61c356213336e646756218539950461ff2bf096badf307a23add6e70053.tar.gz
├── be21f08f670160cbae227e3053205b91d6bfa3de750b90c7e00bd2c511ccb63a.tar.gz
├── manifest.json
└── sha256:bc5d255ea81f83c8c38a982a6d29a6f2198427d258aea5f166e49856896b2da6

0 directories, 5 files

$ jq < nanoserver/manifest.json
[
  {
    "Config": "sha256:bc5d255ea81f83c8c38a982a6d29a6f2198427d258aea5f166e49856896b2da6",
    "RepoTags": [
      "index.docker.io/library/hello-world:i-was-a-digest"
    ],
    "Layers": [
      "a35da61c356213336e646756218539950461ff2bf096badf307a23add6e70053.tar.gz",
      "be21f08f670160cbae227e3053205b91d6bfa3de750b90c7e00bd2c511ccb63a.tar.gz",
      "10d1439be4eb8819987ec2e9c140d44d74d6b42a823d57fe1953bd99948e1bc0.tar.gz"
    ],
    "LayerSources": {
      "sha256:26fd2d9d4c64a4f965bbc77939a454a31b607470f430b5d69fc21ded301fa55e": {
        "mediaType": "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip",
        "size": 101145811,
        "digest": "sha256:a35da61c356213336e646756218539950461ff2bf096badf307a23add6e70053",
        "urls": [
          "https://mcr.microsoft.com/v2/windows/nanoserver/blobs/sha256:a35da61c356213336e646756218539950461ff2bf096badf307a23add6e70053"
        ]
      }
    }
  }
]
```

A couple things to note about this `manifest.json` versus the other:
* The `RepoTags` field is a bit weird here. `hello-world` is a multi-platform
  image, so We had to pull this image by digest, since we're (I'm) on
  amd64/linux and wanted to grab a windows image. Since the tarball format
  expects a tag under `RepoTags`, and we didn't pull by tag, we replace the
  digest with a sentinel `i-was-a-digest` "tag" to appease docker.
* The `LayerSources` has enough information to reconstruct the foreign layers
  pointer when pushing/pulling from the registry. For legal reasons, microsoft
  doesn't want anyone but them to serve windows base images, so the mediaType
  here indicates a "foreign" or "non-distributable" layer with an URL for where
  you can download it from microsoft (see the [OCI
  image-spec](https://github.com/opencontainers/image-spec/blob/master/layer.md#non-distributable-layers)).

We can look at what's in the registry to explain both of these things:
```
$ crane manifest hello-world:nanoserver | jq .
{
  "manifests": [
    {
      "digest": "sha256:63c287625c2b0b72900e562de73c0e381472a83b1b39217aef3856cd398eca0b",
      "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
      "platform": {
        "architecture": "amd64",
        "os": "windows",
        "os.version": "10.0.17763.1040"
      },
      "size": 1124
    }
  ],
  "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
  "schemaVersion": 2
}


# Note the media type and "urls" field.
$ crane manifest hello-world:nanoserver@sha256:63c287625c2b0b72900e562de73c0e381472a83b1b39217aef3856cd398eca0b | jq .
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": {
    "mediaType": "application/vnd.docker.container.image.v1+json",
    "size": 1721,
    "digest": "sha256:bc5d255ea81f83c8c38a982a6d29a6f2198427d258aea5f166e49856896b2da6"
  },
  "layers": [
    {
      "mediaType": "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip",
      "size": 101145811,
      "digest": "sha256:a35da61c356213336e646756218539950461ff2bf096badf307a23add6e70053",
      "urls": [
        "https://mcr.microsoft.com/v2/windows/nanoserver/blobs/sha256:a35da61c356213336e646756218539950461ff2bf096badf307a23add6e70053"
      ]
    },
    {
      "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
      "size": 1669,
      "digest": "sha256:be21f08f670160cbae227e3053205b91d6bfa3de750b90c7e00bd2c511ccb63a"
    },
    {
      "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
      "size": 949,
      "digest": "sha256:10d1439be4eb8819987ec2e9c140d44d74d6b42a823d57fe1953bd99948e1bc0"
    }
  ]
}
```

The `LayerSources` map is keyed by the diffid. Note that `sha256:26fd2d9d4c64a4f965bbc77939a454a31b607470f430b5d69fc21ded301fa55e` matches the first layer in the config file:
```
$ jq '.[0].LayerSources' < nanoserver/manifest.json
{
  "sha256:26fd2d9d4c64a4f965bbc77939a454a31b607470f430b5d69fc21ded301fa55e": {
    "mediaType": "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip",
    "size": 101145811,
    "digest": "sha256:a35da61c356213336e646756218539950461ff2bf096badf307a23add6e70053",
    "urls": [
      "https://mcr.microsoft.com/v2/windows/nanoserver/blobs/sha256:a35da61c356213336e646756218539950461ff2bf096badf307a23add6e70053"
    ]
  }
}

$ jq < nanoserver/sha256\:bc5d255ea81f83c8c38a982a6d29a6f2198427d258aea5f166e49856896b2da6 | jq .rootfs
{
  "type": "layers",
  "diff_ids": [
    "sha256:26fd2d9d4c64a4f965bbc77939a454a31b607470f430b5d69fc21ded301fa55e",
    "sha256:601cf7d78c62e4b4d32a7bbf96a17606a9cea5bd9d22ffa6f34aa431d056b0e8",
    "sha256:a1e1a3bf6529adcce4d91dce2cad86c2604a66b507ccbc4d2239f3da0ec5aab9"
  ]
}
```
