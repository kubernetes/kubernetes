# OCI Image Configuration

An OCI *Image* is an ordered collection of root filesystem changes and the corresponding execution parameters for use within a container runtime.
This specification outlines the JSON format describing images for use with a container runtime and execution tool and its relationship to filesystem changesets, described in [Layers](layer.md).

This section defines the `application/vnd.oci.image.config.v1+json` [media type](media-types.md).

## Terminology

This specification uses the following terms:

### [Layer](layer.md)

* Image filesystems are composed of *layers*.
* Each layer represents a set of filesystem changes in a tar-based [layer format](layer.md), recording files to be added, changed, or deleted relative to its parent layer.
* Layers do not have configuration metadata such as environment variables or default arguments - these are properties of the image as a whole rather than any particular layer.
* Using a layer-based or union filesystem such as AUFS, or by computing the diff from filesystem snapshots, the filesystem changeset can be used to present a series of image layers as if they were one cohesive filesystem.

### Image JSON

* Each image has an associated JSON structure which describes some basic information about the image such as date created, author, as well as execution/runtime configuration like its entrypoint, default arguments, networking, and volumes.
* The JSON structure also references a cryptographic hash of each layer used by the image, and provides history information for those layers.
* This JSON is considered to be immutable, because changing it would change the computed [ImageID](#imageid).
* Changing it means creating a new derived image, instead of changing the existing image.

### Layer DiffID

A layer DiffID is the digest over the layer's uncompressed tar archive and serialized in the descriptor digest format, e.g., `sha256:a9561eb1b190625c9adb5a9513e72c4dedafc1cb2d4c5236c9a6957ec7dfd5a9`.
Layers SHOULD be packed and unpacked reproducibly to avoid changing the layer DiffID, for example by using [tar-split][] to save the tar headers.

NOTE: Do not confuse DiffIDs with [layer digests](manifest.md#image-manifest-property-descriptions), often referenced in the manifest, which are digests over compressed or uncompressed content.

### Layer ChainID

For convenience, it is sometimes useful to refer to a stack of layers with a single identifier.
While a layer's `DiffID` identifies a single changeset, the `ChainID` identifies the subsequent application of those changesets.
This ensures that we have handles referring to both the layer itself, as well as the result of the application of a series of changesets.
Use in combination with `rootfs.diff_ids` while applying layers to a root filesystem to uniquely and safely identify the result.

#### Definition

The `ChainID` of an applied set of layers is defined with the following recursion:

```
ChainID(L₀) =  DiffID(L₀)
ChainID(L₀|...|Lₙ₋₁|Lₙ) = Digest(ChainID(L₀|...|Lₙ₋₁) + " " + DiffID(Lₙ))
```

For this, we define the binary `|` operation to be the result of applying the right operand to the left operand.
For example, given base layer `A` and a changeset `B`, we refer to the result of applying `B` to `A` as `A|B`.

Above, we define the `ChainID` for a single layer (`L₀`) as equivalent to the `DiffID` for that layer.
Otherwise, the `ChainID` for a set of applied layers (`L₀|...|Lₙ₋₁|Lₙ`) is defined as the recursion `Digest(ChainID(L₀|...|Lₙ₋₁) + " " + DiffID(Lₙ))`.

#### Explanation

Let's say we have layers A, B, C, ordered from bottom to top, where A is the base and C is the top.
Defining `|` as a binary application operator, the root filesystem may be `A|B|C`.
While it is implied that `C` is only useful when applied to `A|B`, the identifier `C` is insufficient to identify this result, as we'd have the equality `C = A|B|C`, which isn't true.

The main issue is when we have two definitions of `C`, `C = C` and `C = A|B|C`.
If this is true (with some handwaving), `C = x|C` where `x = any application`.
This means that if an attacker can define `x`, relying on `C` provides no guarantee that the layers were applied in any order.

The `ChainID` addresses this problem by being defined as a compound hash.
__We differentiate the changeset `C`, from the order-dependent application `A|B|C` by saying that the resulting rootfs is identified by ChainID(A|B|C), which can be calculated by `ImageConfig.rootfs`.__

Let's expand the definition of `ChainID(A|B|C)` to explore its internal structure:

```
ChainID(A) = DiffID(A)
ChainID(A|B) = Digest(ChainID(A) + " " + DiffID(B))
ChainID(A|B|C) = Digest(ChainID(A|B) + " " + DiffID(C))
```

We can replace each definition and reduce to a single equality:

```
ChainID(A|B|C) = Digest(Digest(DiffID(A) + " " + DiffID(B)) + " " + DiffID(C))
```

Hopefully, the above is illustrative of the _actual_ contents of the `ChainID`.
Most importantly, we can easily see that `ChainID(C) != ChainID(A|B|C)`, otherwise, `ChainID(C) = DiffID(C)`, which is the base case, could not be true.

### ImageID

Each image's ID is given by the SHA256 hash of its [configuration JSON](#image-json).
It is represented as a hexadecimal encoding of 256 bits, e.g., `sha256:a9561eb1b190625c9adb5a9513e72c4dedafc1cb2d4c5236c9a6957ec7dfd5a9`.
Since the [configuration JSON](#image-json) that gets hashed references hashes of each layer in the image, this formulation of the ImageID makes images content-addressable.

## Properties

Note: Any OPTIONAL field MAY also be set to null, which is equivalent to being absent.

- **created** *string*, OPTIONAL

  An combined date and time at which the image was created, formatted as defined by [RFC 3339, section 5.6][rfc3339-s5.6].

- **author** *string*, OPTIONAL

  Gives the name and/or email address of the person or entity which created and is responsible for maintaining the image.

- **architecture** *string*, REQUIRED

  The CPU architecture which the binaries in this image are built to run on.
  Configurations SHOULD use, and implementations SHOULD understand, values [supported by runtime-spec's `platform.arch`][runtime-platform].

- **os** *string*, REQUIRED

  The name of the operating system which the image is built to run on.
  Configurations SHOULD use, and implementations SHOULD understand, values [supported by runtime-spec's `platform.os`][runtime-platform].

- **config** *object*, OPTIONAL

  The execution parameters which SHOULD be used as a base when running a container using the image.
  This field can be `null`, in which case any execution parameters should be specified at creation of the container.

   - **User** *string*, OPTIONAL

     The username or UID which is a platform-specific structure that allows specific control over which user the process run as.
     This acts as a default value to use when the value is not specified when creating a container.
     For Linux based systems, all of the following are valid: `user`, `uid`, `user:group`, `uid:gid`, `uid:group`, `user:gid`.
     If `group`/`gid` is not specified, the default group and supplementary groups of the given `user`/`uid` in `/etc/passwd` from the container are applied.

   - **ExposedPorts** *object*, OPTIONAL

     A set of ports to expose from a container running this image.
     Its keys can be in the format of:
`port/tcp`, `port/udp`, `port` with the default protocol being `tcp` if not specified.
     These values act as defaults and are merged with any specified when creating a container.
     **NOTE:** This JSON structure value is unusual because it is a direct JSON serialization of the Go type `map[string]struct{}` and is represented in JSON as an object mapping its keys to an empty object.

   - **Env** *array of strings*, OPTIONAL

     Entries are in the format of `VARNAME=VARVALUE`.
     These values act as defaults and are merged with any specified when creating a container.

   - **Entrypoint** *array of strings*, OPTIONAL

     A list of arguments to use as the command to execute when the container starts.
     These values act as defaults and may be replaced by an entrypoint specified when creating a container.

   - **Cmd** *array of strings*, OPTIONAL

     Default arguments to the entrypoint of the container.
     These values act as defaults and may be replaced by any specified when creating a container.
     If an `Entrypoint` value is not specified, then the first entry of the `Cmd` array SHOULD be interpreted as the executable to run.

   - **Volumes** *object*, OPTIONAL

     A set of directories which SHOULD be created as data volumes in a container running this image.
     If a file or folder exists within the image with the same path as a data volume, that file or folder will be replaced by the data volume and never be merged.
     **NOTE:** This JSON structure value is unusual because it is a direct JSON serialization of the Go type `map[string]struct{}` and is represented in JSON as an object mapping its keys to an empty object.

   - **WorkingDir** *string*, OPTIONAL

     Sets the current working directory of the entrypoint process in the container.
     This value acts as a default and may be replaced by a working directory specified when creating a container.

   - **Labels** *object*, OPTIONAL

     The field contains arbitrary metadata for the container.
     This property MUST use the [annotation rules](annotations.md#rules).

  - **StopSignal** *string*, OPTIONAL

    The field contains the system call signal that will be sent to the container to exit. The signal can be a signal name in the format `SIGNAME`, for instance `SIGKILL` or `SIGRTMIN+3`.

- **rootfs** *object*, REQUIRED

   The rootfs key references the layer content addresses used by the image.
   This makes the image config hash depend on the filesystem hash.

    - **type** *string*, REQUIRED

       MUST be set to `layers`.
       Implementations MUST generate an error if they encounter a unknown value while verifying or unpacking an image.

    - **diff_ids** *array of strings*, REQUIRED

       An array of layer content hashes (`DiffIDs`), in order from first to last.

- **history** *array of objects*, OPTIONAL

  Describes the history of each layer.
  The array is ordered from first to last.
  The object has the following fields:

    - **created** *string*, OPTIONAL

       A combined date and time at which the layer was created, formatted as defined by [RFC 3339, section 5.6][rfc3339-s5.6].

    - **author** *string*, OPTIONAL

       The author of the build point.

    - **created_by** *string*, OPTIONAL

       The command which created the layer.

    - **comment** *string*, OPTIONAL

       A custom message set when creating the layer.

    - **empty_layer** *boolean*, OPTIONAL

       This field is used to mark if the history item created a filesystem diff.
       It is set to true if this history item doesn't correspond to an actual layer in the rootfs section (for example, Dockerfile's [ENV](https://docs.docker.com/engine/reference/builder/#/env) command results in no change to the filesystem).

Any extra fields in the Image JSON struct are considered implementation specific and MUST be ignored by any implementations which are unable to interpret them.

Whitespace is OPTIONAL and implementations MAY have compact JSON with no whitespace.

## Example

Here is an example image configuration JSON document:

```json,title=Image%20JSON&mediatype=application/vnd.oci.image.config.v1%2Bjson
{
    "created": "2015-10-31T22:22:56.015925234Z",
    "author": "Alyssa P. Hacker <alyspdev@example.com>",
    "architecture": "amd64",
    "os": "linux",
    "config": {
        "User": "alice",
        "ExposedPorts": {
            "8080/tcp": {}
        },
        "Env": [
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "FOO=oci_is_a",
            "BAR=well_written_spec"
        ],
        "Entrypoint": [
            "/bin/my-app-binary"
        ],
        "Cmd": [
            "--foreground",
            "--config",
            "/etc/my-app.d/default.cfg"
        ],
        "Volumes": {
            "/var/job-result-data": {},
            "/var/log/my-app-logs": {}
        },
        "WorkingDir": "/home/alice",
        "Labels": {
            "com.example.project.git.url": "https://example.com/project.git",
            "com.example.project.git.commit": "45a939b2999782a3f005621a8d0f29aa387e1d6b"
        }
    },
    "rootfs": {
      "diff_ids": [
        "sha256:c6f988f4874bb0add23a778f753c65efe992244e148a1d2ec2a8b664fb66bbd1",
        "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef"
      ],
      "type": "layers"
    },
    "history": [
      {
        "created": "2015-10-31T22:22:54.690851953Z",
        "created_by": "/bin/sh -c #(nop) ADD file:a3bc1e842b69636f9df5256c49c5374fb4eef1e281fe3f282c65fb853ee171c5 in /"
      },
      {
        "created": "2015-10-31T22:22:55.613815829Z",
        "created_by": "/bin/sh -c #(nop) CMD [\"sh\"]",
        "empty_layer": true
      }
    ]
}
```

[rfc3339-s5.6]: https://tools.ietf.org/html/rfc3339#section-5.6
[runtime-platform]: https://github.com/opencontainers/runtime-spec/blob/v1.0.0-rc3/config.md#platform
[tar-split]: https://github.com/vbatts/tar-split
