# `mutate`

[![GoDoc](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/mutate?status.svg)](https://godoc.org/github.com/google/go-containerregistry/pkg/v1/mutate)

The `v1.Image`, `v1.ImageIndex`, and `v1.Layer` interfaces provide only
accessor methods, so they are essentially immutable. If you want to change
something about them, you need to produce a new instance of that interface.

A common use case for this library is to read an image from somewhere (a source),
change something about it, and write the image somewhere else (a sink).

Graphically, this looks something like:

<p align="center">
  <img src="/images/mutate.dot.svg" />
</p>

## Mutations

This is obviously not a comprehensive set of useful transformations (PRs welcome!),
but a rough summary of what the `mutate` package currently does:

### `Config` and `ConfigFile`

These allow you to change the [image configuration](https://github.com/opencontainers/image-spec/blob/master/config.md#properties),
e.g. to change the entrypoint, environment, author, etc.

### `Time`, `Canonical`, and `CreatedAt`

These are useful in the context of [reproducible builds](https://reproducible-builds.org/),
where you may want to strip timestamps and other non-reproducible information.

### `Append`, `AppendLayers`, and `AppendManifests`

These functions allow the extension of a `v1.Image` or `v1.ImageIndex` with
new layers or manifests.

For constructing an image `FROM scratch`, see the [`empty`](/pkg/v1/empty) package.

### `MediaType` and `IndexMediaType`

Sometimes, it is necessary to change the media type of an image or index,
e.g. to appease a registry with strict validation of images (_looking at you, GCR_).

### `Rebase`

Rebase has [its own README](/cmd/crane/rebase.md).

This is the underlying implementation of [`crane rebase`](https://github.com/google/go-containerregistry/blob/main/cmd/crane/doc/crane_rebase.md).

### `Extract`

Extract will flatten an image filesystem into a single tar stream,
respecting whiteout files.

This is the underlying implementation of [`crane export`](https://github.com/google/go-containerregistry/blob/main/cmd/crane/doc/crane_export.md).
