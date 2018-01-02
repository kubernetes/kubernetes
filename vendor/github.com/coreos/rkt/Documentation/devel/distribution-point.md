# Distribution points 

A *distribution point* represents a method for fetching a container image from an input string. This string does not specify an image's type.

A distribution point can provide one or more image formats. Some *Docker* registries also provide *OCI Images*. *Rkt* can fetch a Docker/OCI image from a registry and convert it on the fly to its native image format, ACI. The [docker2aci][docker2aci_GH] tool can perform this conversion in advance.

Before distribution points, rkt used *ImageTypes*. These mapped a specifically formatted input string to things like the distribution, transport and image type. This information is hidden now since all images are appc ACIs.

Distribution points are used as the primary expression of container image information in the different layers of rkt.  This includes fetching and referencing in a CAS/ref store.

## Distribution points types

Distribution points are either direct or indirect. Direct distribution points provide the final information needed to fetch the image. Indirect distribution points take some indirect steps, like discovery, before getting the final image location. An indirect distribution point may resolve to a direct distribution point.

## Distribution points format

A distribution point is represented as a URI with the URI scheme as "cimd" and the remaining parts (URI opaque data and query/fragments parts) as the distribution point data. See [rfc3986][3986] for more information on this. Distribution points clearly map to a resource name, otherwise they will not fit inside a resource locator (URL). We will then use the term URIs instead of URNs because it's the suggested name from the rfc (and URNs are defined, by rfc2141, to have the `urn` scheme).

Every distribution starts the same: `cimd:DISTTYPE:v=uint32(VERSION):` where

* `cimd` is the *container image distribution scheme*
* `DISTTYPE` is the *distribution type*
* `v=uint32(VERSION)` is the *distribution type format version*

## Current rkt distribution points

Rkt has three types of distribution points:

* `Appc`
* `ACIArchive`
* `Docker`

### Appc

This is an indirect distribution point.

Appc defines a distribution point using appc image discovery

* The format is: `cimd:appc:v=0:name?label01=....&label02=....`
* The distribution type is "appc"
* The labels values must be Query escaped

**Example:** `cimd:appc:v=0:coreos.com/etcd?version=v3.0.3&os=linux&arch=amd64`

### ACIArchive

This is a direct distribution point since it directly define the final image location.

ACIArchive defines a distribution point using an archive file

* The format is: `cimd:aci-archive:v=0:ArchiveURL?query...`
* The distribution type is "aci-archive"
* ArchiveURL must be query escaped

**Examples**:

* `cimd:aci-archive:v=0:file%3A%2F%2Fabsolute%2Fpath%2Fto%2Ffile`
* `cimd:aci-archive:v=0:https%3A%2F%2Fexample.com%2Fapp.aci`

### Docker

Docker is an indirect distribution point.

This defines a distribution point using a docker registry

The format is:

* `cimd:docker:v=0:[REGISTRY_HOST[:REGISTRY_PORT]/]NAME[:TAG|@DIGEST]`
* Removing the common distribution point section, the format is the same as the docker image string format (man docker-pull).

**Examples**:

* `cimd:docker:v=0:busybox`
* `cimd:docker:v=0:busybox:latest`
* `cimd:docker:v=0:registry-1.docker.io/library/busybox@sha256:a59906e33509d14c036c8678d687bd4eec81ed7c4b8ce907b888c607f6a1e0e6`

### Future distribution points

#### OCI Image distribution(s)

This is an Indirect distribution point.

OCI images can be retrieved using a Docker registry but in future the OCI image spec will define one or more own kinds of distribution starting from an image name (with additional tags/labels).

#### OCI Image layout

This is a Direct distribution point.

This can fetch an image starting from a [OCI image layout][oci_layout] format. The 'location' can point to:

* A single file archive
* A local directory based layout
* A remote directory based layout
* Other types of locations

This will probably end up being the final distribution used by the above OCI image distributions (like ACIArchive is the final distribution point for the Appc distribution point):

* `cimd:oci-image-layout:v=0:file%3A%2F%2Fabsolute%2Fpath%2Fto%2Ffile?ref=refname`
* `cimd:oci-image-layout:v=0:https%3A%2F%2Fdir%2F?ref=refname`

Since the OCI image layout can provide multiple images selectable by a ref, one needs to specify which ref to use in the archive distribution URI (see the above ref query parameter). Since distribution only covers one image, it is not possible to import all refs with a single distribution URI.

**TODO(sgotti)**: Define if oci-image-layout. It should internally handle both archive and directory based layouts or use two different distributions or a query parameter the explicitly define the layout (to avoid guessing if the URL points to a single file or to a directory).*

**Note** Considering [this OCI image spec README section][oci_image_spec_readme], the final distribution format will probably be similar to the Appc distribution. There is a need to distinguish their User Friendly string (prepending an appc: or oci: ?).

To sum it up:

| Distribution Point | Type     | URI Format                                                                | Final Distribution Point |
|--------------------|----------|---------------------------------------------------------------------------|--------------------------|
| Appc               | Direct   | `cimd:appc:v=0:name?label01=....&label02=...`                             | ACIArchive               |
| Docker             | Direct   | `cimd:docker:v=0:[REGISTRY_HOST[:REGISTRY_PORT]/]NAME[:TAG&#124;@DIGEST]` | <none>                   |
| ACIArchive         | Indirect | `cimd:aci-archive:v=0:ArchiveURL?query...`                                |                          |
| OCI                | Direct   | `cimd:oci:v=0:TODO`                                                       | OCIImageLayout           |
| OCIImageLayout     | Indirect | `cimd:oci-image-layout:v=0:URL?ref=...`                                   |                          |

## User-friendly distribution strings

The distribution URI can be long and complex. It is helpful to have a friendly string for users to request an image with. Rkt supports a couple of image string input styles. These are mapped to an `AppImageType`:

* Appc discovery string: `example.com/app01:v1.0.0,label01=value01,...` or `example.com/app01,version=v1.0.0,label01=value01,...` etc.
* File paths are absolute (`/full/path/to/file`) or relative.

The above two may overlap so some heuristic is needed to distinguish them (removing this heuristic will break backward compatibility in the CLI).

* File URL: `file:///full/path/to/file`
* Http(s) URL: `http(s)://host:port/path`
* Docker URL: This is a strange URL since it the schemeful (`docker://`) version of the docker image string

To maintain backward compatibility these image string will be converted to a distribution URI:

| Current ImageType                      | Distribution Point URI                                                          |
|----------------------------------------|---------------------------------------------------------------------------|
| appc string                            | `cimd:appc:v=0:name?label01=....&label02=...`                             |
| file path                              | `cimd:aci-archive:v=0:ArchiveURL`                                         |
| file URL                               | `cimd:aci-archive:v=0:ArchiveURL`                                         |
| https URL                              | `cimd:aci-archive:v=0:ArchiveURL`                                         |
| docker URI/URL (docker: and docker://) | `cimd:docker:v=0:[REGISTRY_HOST[:REGISTRY_PORT]/]NAME[:TAG&#124;@DIGEST]` |

The above table also adds Docker URI (`docker:`) as a user friendly string and its clearer than the URL version (`docker://`)

The parsing and generation of user friendly string is done outside the distribution package (to let distribution pkg users implement their own user friendly strings).

Rkt has two jobs:

* Parse a user friendly string to a distribution URI.
* Generate a user friendly string from a distribution URI. This is useful when showing the refs from a refs store. They can easily be understood and copy/pasted.

A user can provide as an input image as a "user friendly" string or a complete distribution URI.

## Comparing Distribution Points URIs

A Distribution Point implementation will also provide a function to compare if Distribution Point URIs are the same (e.g. ordering the query parameters).

## Fetching logic with Distribution Points

A Distribution Point will be the base for a future refactor of the fetchers logic (see [#2964][rkt-2964])

This also creates a better separation between the distribution points and the transport layers.

For example there may exist multiple transport plugins (file, http, s3, bittorrent etc...) to be called by an ACIArchive distribution point.


[3986]: https://tools.ietf.org/html/rfc3986
[docker2aci_GH]: https://github.com/appc/docker2aci
[oci_image_spec_readme]: https://github.com/opencontainers/image-spec#running-an-oci-imag://github.com/appc/docker2aci
[oci_layout]: https://github.com/opencontainers/image-spec/blob/master/image-layout.md
[rkt-2964]: https://github.com/coreos/rkt/pull/2964
