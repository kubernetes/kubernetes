# Annotations
Several components of the specification, like [Image Manifests](manifest.md) and [Descriptors](descriptor.md), feature an optional annotations property, whose format is common and defined in this section.

This property contains arbitrary metadata.

## Rules

* Annotations MUST be a key-value map where both the key and value MUST be strings.
* While the value MUST be present, it MAY be an empty string.
* Keys MUST be unique within this map, and best practice is to namespace the keys.
* Keys SHOULD be named using a reverse domain notation - e.g. `com.example.myKey`.
* The prefix `org.opencontainers` is reserved for keys defined in Open Container Initiative (OCI) specifications and MUST NOT be used by other specifications and extensions.
* Keys using the `org.opencontainers.image` namespace are reserved for use in the OCI Image Specification and MUST NOT be used by other specifications and extensions, including other OCI specifications.
* If there are no annotations then this property MUST either be absent or be an empty map.
* Consumers MUST NOT generate an error if they encounter an unknown annotation key.

## Pre-Defined Annotation Keys

This specification defines the following annotation keys, intended for but not limited to [image index](image-index.md) and image [manifest](manifest.md) authors:
* **org.opencontainers.image.created** date and time on which the image was built (string, date-time as defined by [RFC 3339](https://tools.ietf.org/html/rfc3339#section-5.6)).
* **org.opencontainers.image.authors** contact details of the people or organization responsible for the image (freeform string)
* **org.opencontainers.image.url** URL to find more information on the image (string)
* **org.opencontainers.image.documentation** URL to get documentation on the image (string)
* **org.opencontainers.image.source** URL to get source code for building the image (string)
* **org.opencontainers.image.version** version of the packaged software
  * The version MAY match a label or tag in the source code repository
  * version MAY be [Semantic versioning-compatible](http://semver.org/)
* **org.opencontainers.image.revision** Source control revision identifier for the packaged software.
* **org.opencontainers.image.vendor** Name of the distributing entity, organization or individual.
* **org.opencontainers.image.licenses** License(s) under which contained software is distributed as an [SPDX License Expression][spdx-license-expression].
* **org.opencontainers.image.ref.name** Name of the reference for a target (string). SHOULD only be considered valid when on descriptors on `index.json` within [image layout](image-layout.md).
* **org.opencontainers.image.title** Human-readable title of the image (string)
* **org.opencontainers.image.description** Human-readable description of the software packaged in the image (string)

## Back-compatibility with Label Schema

[Label Schema](https://label-schema.org) defined a number of conventional labels for container images, and these are now superceded by annotations with keys starting **org.opencontainers.image**.

While users are encouraged to use the **org.opencontainers.image** keys, tools MAY choose to support compatible annotations using the **org.label-schema** prefix as follows.

| `org.opencontainers.image` prefix | `org.label-schema prefix` | Compatibility notes |
|---------------------------|-------------------------|---------------------|
| `created` | `build-date` | Compatible |
| `url` | `url` | Compatible |
| `source` | `vcs-url` | Compatible |
| `version` | `version` | Compatible |
| `revision` | `vcs-ref` | Compatible |
| `vendor` | `vendor` | Compatible |
| `title` | `name` | Compatible |
| `description` | `description` | Compatible |
| `documentation` | `usage` | Value is compatible if the documentation is located by a URL |
| `authors` |  | No equivalent in Label Schema |
| `licenses` | | No equivalent in Label Schema |
| `ref.name` | | No equivalent in Label Schema |
| | `schema-version`| No equivalent in the OCI Image Spec |
| | `docker.*`, `rkt.*` | No equivalent in the OCI Image Spec |

[spdx-license-expression]: https://spdx.org/spdx-specification-21-web-version#h.jxpfx0ykyb60
