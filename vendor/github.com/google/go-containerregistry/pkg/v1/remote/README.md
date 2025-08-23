# `remote`

[![GoDoc](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/remote?status.svg)](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/remote)

The `remote` package implements a client for accessing a registry,
per the [OCI distribution spec](https://github.com/opencontainers/distribution-spec/blob/master/spec.md).

It leans heavily on the lower level [`transport`](/pkg/v1/remote/transport) package, which handles the
authentication handshake and structured errors.

## Usage

```go
package main

import (
	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote"
)

func main() {
	ref, err := name.ParseReference("gcr.io/google-containers/pause")
	if err != nil {
		panic(err)
	}

	img, err := remote.Image(ref, remote.WithAuthFromKeychain(authn.DefaultKeychain))
	if err != nil {
		panic(err)
	}

	// do stuff with img
}
```

## Structure

<p align="center">
  <img src="/images/remote.dot.svg" />
</p>


## Background

There are a lot of confusingly similar terms that come up when talking about images in registries.

### Anatomy of an image

In general...

* A tag refers to an image manifest.
* An image manifest references a config file and an orderered list of _compressed_ layers by sha256 digest.
* A config file references an ordered list of _uncompressed_ layers by sha256 digest and contains runtime configuration.
* The sha256 digest of the config file is the [image id](https://github.com/opencontainers/image-spec/blob/master/config.md#imageid) for the image.

For example, an image with two layers would look something like this:

![image anatomy](/images/image-anatomy.dot.svg)

### Anatomy of an index

In the normal case, an [index](https://github.com/opencontainers/image-spec/blob/master/image-index.md) is used to represent a multi-platform image.
This was the original use case for a [manifest
list](https://docs.docker.com/registry/spec/manifest-v2-2/#manifest-list).

![image index anatomy](/images/index-anatomy.dot.svg)

It is possible for an index to reference another index, per the OCI
[image-spec](https://github.com/opencontainers/image-spec/blob/master/media-types.md#compatibility-matrix).
In theory, both an image and image index can reference arbitrary things via
[descriptors](https://github.com/opencontainers/image-spec/blob/master/descriptor.md),
e.g. see the [image layout
example](https://github.com/opencontainers/image-spec/blob/master/image-layout.md#index-example),
which references an application/xml file from an image index.

That could look something like this:

![strange image index anatomy](/images/index-anatomy-strange.dot.svg)

Using a recursive index like this might not be possible with all registries,
but this flexibility allows for some interesting applications, e.g. the
[OCI Artifacts](https://github.com/opencontainers/artifacts) effort.

### Anatomy of an image upload

The structure of an image requires a delicate ordering when uploading an image to a registry.
Below is a (slightly simplified) figure that describes how an image is prepared for upload
to a registry and how the data flows between various artifacts:

![upload](/images/upload.dot.svg)

Note that:

* A config file references the uncompressed layer contents by sha256.
* A manifest references the compressed layer contents by sha256 and the size of the layer.
* A manifest references the config file contents by sha256 and the size of the file.

It follows that during an upload, we need to upload layers before the config file,
and we need to upload the config file before the manifest.

Sometimes, we know all of this information ahead of time, (e.g. when copying from remote.Image),
so the ordering is less important.

In other cases, e.g. when using a [`stream.Layer`](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/stream#Layer),
we can't compute anything until we have already uploaded the layer, so we need to be careful about ordering.

## Caveats

### schema 1

This package does not support schema 1 images, see [`#377`](https://github.com/google/go-containerregistry/issues/377),
however, it's possible to do _something_ useful with them via [`remote.Get`](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/remote#Get),
which doesn't try to interpret what is returned by the registry.

[`crane.Copy`](https://godoc.org/github.com/google/go-containerregistry/pkg/crane#Copy) takes advantage of this to implement support for copying schema 1 images,
see [here](https://github.com/google/go-containerregistry/blob/main/pkg/internal/legacy/copy.go).
