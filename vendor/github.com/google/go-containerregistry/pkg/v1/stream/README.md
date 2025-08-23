# `stream`

[![GoDoc](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/stream?status.svg)](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/stream)

The `stream` package contains an implementation of
[`v1.Layer`](https://godoc.org/github.com/google/go-containerregistry/pkg/v1#Layer)
that supports _streaming_ access, i.e. the layer contents are read once and not
buffered.

## Usage

```go
package main

import (
	"os"

	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote"
	"github.com/google/go-containerregistry/pkg/v1/stream"
)

// upload the contents of stdin as a layer to a local registry
func main() {
	repo, err := name.NewRepository("localhost:5000/stream")
	if err != nil {
		panic(err)
	}

	layer := stream.NewLayer(os.Stdin)

	if err := remote.WriteLayer(repo, layer); err != nil {
		panic(err)
	}
}
```

## Structure

This implements the layer portion of an [image
upload](/pkg/v1/remote#anatomy-of-an-image-upload). We launch a goroutine that
is responsible for hashing the uncompressed contents to compute the `DiffID`,
gzipping them to produce the `Compressed` contents, and hashing/counting the
bytes to produce the `Digest`/`Size`. This goroutine writes to an
`io.PipeWriter`, which blocks until `Compressed` reads the gzipped contents from
the corresponding `io.PipeReader`.

<p align="center">
  <img src="/images/stream.dot.svg" />
</p>

## Caveats

This assumes that you have an uncompressed layer (i.e. a tarball) and would like
to compress it. Calling `Uncompressed` is always an error. Likewise, other
methods are invalid until the contents of `Compressed` have been completely
consumed and `Close`d.

Using a `stream.Layer` will likely not work without careful consideration. For
example, in the `mutate` package, we defer computing the manifest and config
file until they are actually called. This allows you to `mutate.Append` a
streaming layer to an image without accidentally consuming it. Similarly, in
`remote.Write`, if calling `Digest` on a layer fails, we attempt to upload the
layer anyway, understanding that we may be dealing with a `stream.Layer` whose
contents need to be uploaded before we can upload the config file.

Given the [structure](#structure) of how this is implemented, forgetting to
`Close` a `stream.Layer` will leak a goroutine.
