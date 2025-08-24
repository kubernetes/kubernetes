# `empty`

[![GoDoc](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/empty?status.svg)](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/empty)

The empty packages provides an empty base for constructing a `v1.Image` or `v1.ImageIndex`.
This is especially useful when paired with the [`mutate`](/pkg/v1/mutate) package,
see [`mutate.Append`](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/mutate#Append)
and [`mutate.AppendManifests`](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/mutate#AppendManifests).
