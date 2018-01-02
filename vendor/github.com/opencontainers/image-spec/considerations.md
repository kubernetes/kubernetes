# Extensibility

Implementations that are reading/processing [manifests](manifest.md) or [image indexes](image-index.md) MUST NOT generate an error if they encounter an unknown property.
Instead they MUST ignore unknown properties.

# Canonicalization

* OCI Images are [content-addressable](https://en.wikipedia.org/wiki/Content-addressable_storage). See [descriptors](descriptor.md) for more.
* One benefit of content-addressable storage is easy deduplication.
* Many images might depend on a particular [layer](layer.md), but there will only be one blob in the [store](image-layout.md).
* With a different serialization, that same semantic layer would have a different hash, and if both versions of the layer are referenced there will be two blobs with the same semantic content.
* To allow efficient storage, implementations serializing content for blobs SHOULD use a canonical serialization.
* This increases the chance that different implementations can push the same semantic content to the store without creating redundant blobs.

## JSON

[JSON][] content SHOULD be serialized as [canonical JSON][canonical-json].
Of the [OCI Image Format Specification media types](media-types.md), all the types ending in `+json` contain JSON content.
Implementations:

* [Go][]: [github.com/docker/go][], which claims to implement [canonical JSON][canonical-json] except for Unicode normalization.

[canonical-json]: http://wiki.laptop.org/go/Canonical_JSON
[github.com/docker/go]: https://github.com/docker/go/
[Go]: https://golang.org/
[JSON]: http://json.org/
