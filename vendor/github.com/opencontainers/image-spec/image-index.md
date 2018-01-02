# OCI Image Index Specification

The image index is a higher-level manifest which points to specific [image manifests](manifest.md), ideal for one or more platforms.
While the use of an image index is OPTIONAL for image providers, image consumers SHOULD be prepared to process them.

This section defines the `application/vnd.oci.image.index.v1+json` [media type](media-types.md).
For the media type(s) that this document is compatible with, see the [matrix][matrix].

## *Image Index* Property Descriptions

- **`schemaVersion`** *int*

  This REQUIRED property specifies the image manifest schema version.
  For this version of the specification, this MUST be `2` to ensure backward compatibility with older versions of Docker.
  The value of this field will not change.
  This field MAY be removed in a future version of the specification.

- **`mediaType`** *string*

  This property is *reserved* for use, to [maintain compatibility][matrix].
  When used, this field contains the media type of this document, which differs from the [descriptor](descriptor.md#properties) use of `mediaType`.

- **`manifests`** *array of objects*

  This REQUIRED property contains a list of [manifests](manifest.md) for specific platforms.
  While this property MUST be present, the size of the array MAY be zero.

  Each object in `manifests` includes a set of [descriptor properties](descriptor.md#properties) with the following additional properties and restrictions:

  - **`mediaType`** *string*

    This [descriptor property](descriptor.md#properties) has additional restrictions for `manifests`.
    Implementations MUST support at least the following media types:

    - [`application/vnd.oci.image.manifest.v1+json`](manifest.md)

    Image indexes concerned with portability SHOULD use one of the above media types.
    Future versions of the spec MAY use a different mediatype (i.e. a new versioned format).
    An encountered `mediaType` that is unknown SHOULD be safely ignored.

  - **`platform`** *object*

    This OPTIONAL property describes the minimum runtime requirements of the image.
    This property SHOULD be present if its target is platform-specific.

    - **`architecture`** *string*

        This REQUIRED property specifies the CPU architecture.
        Image indexes SHOULD use, and implementations SHOULD understand, values [supported by runtime-spec's `platform.arch`][runtime-platform2].

    - **`os`** *string*

        This REQUIRED property specifies the operating system.
        Image indexes SHOULD use, and implementations SHOULD understand, values [supported by runtime-spec's `platform.os`][runtime-platform2].

    - **`os.version`** *string*

        This OPTIONAL property specifies the version of the operating system targeted by the referenced blob.
        Implementations MAY refuse to use manifests where `os.version` is not known to work with the host OS version.
        Valid values are implementation-defined. e.g. `10.0.14393.1066` on `windows`.

    - **`os.features`** *array of strings*

        This OPTIONAL property specifies an array of strings, each specifying a mandatory OS feature (for example on Windows `win32k`).

    - **`variant`** *string*

        This OPTIONAL property specifies the variant of the CPU.
        Image indexes SHOULD use, and implementations SHOULD understand, values listed in the following table.
        When the variant of the CPU is not listed in the table, values are implementation-defined and SHOULD be submitted to this specification for standardization.

        | ISA/ABI         | `architecture` | `variant`   |
        |-----------------|----------------|-------------|
        | ARM 32-bit, v6  | `arm`          | `v6`        |
        | ARM 32-bit, v7  | `arm`          | `v7`        |
        | ARM 32-bit, v8  | `arm`          | `v8`        |
        | ARM 64-bit, v8  | `arm64`        | `v8`        |

    - **`features`** *array of strings*

        This property is RESERVED for future versions of the specification.

- **`annotations`** *string-string map*

    This OPTIONAL property contains arbitrary metadata for the image index.
    This OPTIONAL property MUST use the [annotation rules](annotations.md#rules).

    See [Pre-Defined Annotation Keys](annotations.md#pre-defined-annotation-keys).

## Example Image Index

*Example showing a simple image index pointing to image manifests for two platforms:*
```json,title=Image%20Index&mediatype=application/vnd.oci.image.index.v1%2Bjson
{
  "schemaVersion": 2,
  "manifests": [
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "size": 7143,
      "digest": "sha256:e692418e4cbaf90ca69d05a66403747baa33ee08806650b51fab815ad7fc331f",
      "platform": {
        "architecture": "ppc64le",
        "os": "linux"
      }
    },
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "size": 7682,
      "digest": "sha256:5b0bcabd1ed22e9fb1310cf6c2dec7cdef19f0ad69efa1f392e94a4333501270",
      "platform": {
        "architecture": "amd64",
        "os": "linux"
      }
    }
  ],
  "annotations": {
    "com.example.key1": "value1",
    "com.example.key2": "value2"
  }
}
```

[runtime-platform2]: https://github.com/opencontainers/runtime-spec/blob/v1.0.0-rc3/config.md#platform
[matrix]: media-types.md#compatibility-matrix
