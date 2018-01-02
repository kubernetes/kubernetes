# OCI Content Descriptors

* An OCI image consists of several different components, arranged in a [Merkle Directed Acyclic Graph (DAG)](https://en.wikipedia.org/wiki/Merkle_tree).
* References between components in the graph are expressed through _Content Descriptors_.
* A Content Descriptor (or simply _Descriptor_) describes the disposition of the targeted content.
* A Content Descriptor includes the type of the content, a content identifier (_digest_), and the byte-size of the raw content.
* Descriptors SHOULD be embedded in other formats to securely reference external content.
* Other formats SHOULD use descriptors to securely reference external content.

This section defines the `application/vnd.oci.descriptor.v1+json` [media type](media-types.md).

## Properties

A descriptor consists of a set of properties encapsulated in key-value fields.

The following fields contain the primary properties that constitute a Descriptor:

- **`mediaType`** *string*

  This REQUIRED property contains the media type of the referenced content.
  Values MUST comply with [RFC 6838][rfc6838], including the [naming requirements in its section 4.2][rfc6838-s4.2].

  The OCI image specification defines [several of its own MIME types](media-types.md) for resources defined in the specification.

- **`digest`** *string*

  This REQUIRED property is the _digest_ of the targeted content, conforming to the requirements outlined in [Digests and Verification](#digests-and-verification).
  Retrieved content SHOULD be verified against this digest when consumed via untrusted sources.

- **`size`** *int64*

  This REQUIRED property specifies the size, in bytes, of the raw content.
  This property exists so that a client will have an expected size for the content before processing.
  If the length of the retrieved content does not match the specified length, the content SHOULD NOT be trusted.

- **`urls`** *array of strings*

  This OPTIONAL property specifies a list of URIs from which this object MAY be downloaded.
  Each entry MUST conform to [RFC 3986][rfc3986].
  Entries SHOULD use the `http` and `https` schemes, as defined in [RFC 7230][rfc7230-s2.7].

- **`annotations`** *string-string map*

    This OPTIONAL property contains arbitrary metadata for this descriptor.
    This OPTIONAL property MUST use the [annotation rules](annotations.md#rules).

Descriptors pointing to [`application/vnd.oci.image.manifest.v1+json`](manifest.md) SHOULD include the extended field `platform`, see [Image Index Property Descriptions](image-index.md#image-index-property-descriptions) for details.

### Reserved

The following field keys are reserved and MUST NOT be used by other specifications.

- **`data`** *string*

  This key is RESERVED for future versions of the specification.

All other fields may be included in other OCI specifications.
Extended _Descriptor_ field additions proposed in other OCI specifications SHOULD first be considered for addition into this specification.

## Digests

The _digest_ property of a Descriptor acts as a content identifier, enabling [content addressability](http://en.wikipedia.org/wiki/Content-addressable_storage).
It uniquely identifies content by taking a [collision-resistant hash](https://en.wikipedia.org/wiki/Cryptographic_hash_function) of the bytes.
If the _digest_ can be communicated in a secure manner, one can verify content from an insecure source by recalculating the digest independently, ensuring the content has not been modified.

The value of the `digest` property is a string consisting of an _algorithm_ portion and an _encoded_ portion.
The _algorithm_ specifies the cryptographic hash function and encoding used for the digest; the _encoded_ portion contains the encoded result of the hash function.

A digest string MUST match the following grammar:

```
digest                := algorithm ":" encoded
algorithm             := algorithm-component [algorithm-separator algorithm-component]*
algorithm-component   := /[a-z0-9]+/
algorithm-separator   := /[+._-]/
encoded               := /[a-zA-Z0-9=_-]+/
```

Note that _algorithm_ MAY impose algorithm-specific restriction on the grammar of the _encoded_ portion.
See also [Registered Algorithms](#registered-algorithms).

Some example digest strings include the following:

digest                                                                    | algorithm           | Registered |
--------------------------------------------------------------------------|---------------------|------------|
`sha256:6c3c624b58dbbcd3c0dd82b4c53f04194d1247c6eebdaab7c610cf7d66709b3b` | [SHA-256](#sha-256) | Yes        |
`sha512:401b09eab3c013d4ca54922bb802bec8fd5318192b0a75f201d8b372742...`   | [SHA-512](#sha-512) | Yes        |
`multihash+base58:QmRZxt2b1FVZPNqd8hsiykDL3TdBDeTSPX9Kv46HmX4Gx8`         | Multihash           | No         |
`sha256+b64u:LCa0a2j_xo_5m0U8HTBBNBNCLXBkg7-g-YpeiGJm564`                 | SHA-256 with urlsafe base64 | No |

Please see [Registered Algorithms](#registered-algorithms) for a list of registered algorithms.

Implementations SHOULD allow digests with unrecognized algorithms to pass validation if they comply with the above grammar.
While `sha256` will only use hex encoded digests, separators in _algorithm_ and alphanumerics in _encoded_ are included to allow for extensions.
As an example, we can parameterize the encoding and algorithm as `multihash+base58:QmRZxt2b1FVZPNqd8hsiykDL3TdBDeTSPX9Kv46HmX4Gx8`, which would be considered valid but unregistered by this specification.

### Verification

Before consuming content targeted by a descriptor from untrusted sources, the byte content SHOULD be verified against the digest string.
Before calculating the digest, the size of the content SHOULD be verified to reduce hash collision space.
Heavy processing before calculating a hash SHOULD be avoided.
Implementations MAY employ [canonicalization](canonicalization.md#canonicalization) of the underlying content to ensure stable content identifiers.

### Digest calculations

A _digest_ is calculated by the following pseudo-code, where `H` is the selected hash algorithm, identified by string `<alg>`:
```
let ID(C) = Descriptor.digest
let C = <bytes>
let D = '<alg>:' + Encode(H(C))
let verified = ID(C) == D
```
Above, we define the content identifier as `ID(C)`, extracted from the `Descriptor.digest` field.
Content `C` is a string of bytes.
Function `H` returns the hash of `C` in bytes and is passed to function `Encode` and prefixed with the algorithm to obtain the digest.
The result `verified` is true if `ID(C)` is equal to `D`, confirming that `C` is the content identified by `D`.
After verification, the following is true:

```
D == ID(C) == '<alg>:' + Encode(H(C))
```

The _digest_ is confirmed as the content identifier by independently calculating the _digest_.

### Registered algorithms

While the _algorithm_ component of the digest string allows the use of a variety of cryptographic algorithms, compliant implementations SHOULD use [SHA-256](#sha-256).

The following algorithm identifiers are currently defined by this specification:

| algorithm identifier | algorithm           |
|----------------------|---------------------|
| `sha256`             | [SHA-256](#sha-256) |
| `sha512`             | [SHA-512](#sha-512) |

If a useful algorithm is not included in the above table, it SHOULD be submitted to this specification for registration.

#### SHA-256

[SHA-256][rfc4634-s4.1] is a collision-resistant hash function, chosen for ubiquity, reasonable size and secure characteristics.
Implementations MUST implement SHA-256 digest verification for use in descriptors.

When the _algorithm identifier_ is `sha256`, the _encoded_ portion MUST match `/[a-f0-9]{64}/`.
Note that `[A-F]` MUST NOT be used here.

#### SHA-512

[SHA-512][rfc4634-s4.2] is a collision-resistant hash function which [may be more perfomant][sha256-vs-sha512] than [SHA-256](#sha-256) on some CPUs.
Implementations MAY implement SHA-512 digest verification for use in descriptors.

When the _algorithm identifier_ is `sha512`, the _encoded_ portion MUST match `/[a-f0-9]{128}/`.
Note that `[A-F]` MUST NOT be used here.

## Examples

The following example describes a [_Manifest_](manifest.md#image-manifest) with a content identifier of "sha256:5b0bcabd1ed22e9fb1310cf6c2dec7cdef19f0ad69efa1f392e94a4333501270" and a size of 7682 bytes:

```json,title=Content%20Descriptor&mediatype=application/vnd.oci.descriptor.v1%2Bjson
{
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "size": 7682,
  "digest": "sha256:5b0bcabd1ed22e9fb1310cf6c2dec7cdef19f0ad69efa1f392e94a4333501270"
}
```

In the following example, the descriptor indicates that the referenced manifest is retrievable from a particular URL:

```json,title=Content%20Descriptor&mediatype=application/vnd.oci.descriptor.v1%2Bjson
{
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "size": 7682,
  "digest": "sha256:5b0bcabd1ed22e9fb1310cf6c2dec7cdef19f0ad69efa1f392e94a4333501270",
  "urls": [
    "https://example.com/example-manifest"
  ]
}
```

[rfc3986]: https://tools.ietf.org/html/rfc3986
[rfc4634-s4.1]: https://tools.ietf.org/html/rfc4634#section-4.1
[rfc4634-s4.2]: https://tools.ietf.org/html/rfc4634#section-4.2
[rfc6838]: https://tools.ietf.org/html/rfc6838
[rfc6838-s4.2]: https://tools.ietf.org/html/rfc6838#section-4.2
[rfc7230-s2.7]: https://tools.ietf.org/html/rfc7230#section-2.7
[sha256-vs-sha512]: https://groups.google.com/a/opencontainers.org/forum/#!topic/dev/hsMw7cAwrZE
