# Conversion to OCI Runtime Configuration

When extracting an OCI Image into an [OCI Runtime bundle][oci-runtime-bundle], two orthogonal components of the extraction are relevant:

1. Extraction of the root filesystem from the set of [filesystem layers](layer.md).
2. Conversion of the [image configuration blob](config.md) to an [OCI Runtime configuration blob][oci-runtime-config].

This section defines how to convert an `application/vnd.oci.image.config.v1+json` blob to an [OCI runtime configuration blob][oci-runtime-config] (the latter component of extraction).
The former component of extraction is defined [elsewhere](layer.md) and is orthogonal to configuration of a runtime bundle.
The values of runtime configuration properties not specified by this document are implementation-defined.

A converter MUST rely on the OCI image configuration to build the OCI runtime configuration as described by this document; this will create the "default generated runtime configuration".

The "default generated runtime configuration" MAY be overridden or combined with externally provided inputs from the caller.
In addition, a converter MAY have its own implementation-defined defaults and extensions which MAY be combined with the "default generated runtime configuration".
The restrictions in this document refer only to combining implementation-defined defaults with the "default generated runtime configuration".
Externally provided inputs are considered to be a modification of the `application/vnd.oci.image.config.v1+json` used as a source, and such modifications have no restrictions.

For example, externally provided inputs MAY cause an environment variable to be added, removed or changed.
However an implementation-defined default SHOULD NOT result in an environment variable being removed or changed.

[oci-runtime-bundle]: https://github.com/opencontainers/runtime-spec/blob/v1.0.0-rc5/bundle.md
[oci-runtime-config]: https://github.com/opencontainers/runtime-spec/blob/v1.0.0-rc5/config.md

## Verbatim Fields

Certain image configuration fields have an identical counterpart in the runtime configuration.
Some of these are purely annotation-based fields, and have been extracted into a [separate subsection](#annotation-fields).
A compliant configuration converter MUST extract the following fields verbatim to the corresponding field in the generated runtime configuration:

| Image Field         | Runtime Field   | Notes |
| ------------------- | --------------- | ----- |
| `architecture`      | `platform.arch` |       |
| `os`                | `platform.os`   |       |
| `Config.WorkingDir` | `process.cwd`   |       |
| `Config.Env`        | `process.env`   | 1     |
| `Config.Entrypoint` | `process.args`  | 2     |
| `Config.Cmd`        | `process.args`  | 2     |

1. The converter MAY add additional entries to `process.env` but it SHOULD NOT add entries that have variable names present in `Config.Env`.
2. If both `Config.Entrypoint` and `Config.Cmd` are specified, the converter MUST append the value of `Config.Cmd` to the value of `Config.Entrypoint` and set `process.args` to that combined value.

### Annotation Fields

These fields all affect the `annotations` of the runtime configuration, and are thus subject to [precedence](#annotations).

| Image Field         | Runtime Field   | Notes |
| ------------------- | --------------- | ----- |
| `author`            | `annotations`   | 1,2   |
| `created`           | `annotations`   | 1,3   |
| `Config.Labels`     | `annotations`   |       |
| `Config.StopSignal` | `annotations`   | 1,4   |

1. If a user has explicitly specified this annotation with `Config.Labels`, then the value specified in this field takes lower [precedence](#annotations) and the converter MUST instead use the value from `Config.Labels`.
2. The value of this field MUST be set as the value of `org.opencontainers.image.author` in `annotations`.
3. The value of this field MUST be set as the value of `org.opencontainers.image.created` in `annotations`.
4. The value of this field MUST be set as the value of `org.opencontainers.image.stopSignal` in `annotations`.

## Parsed Fields

Certain image configuration fields have a counterpart that must first be translated.
A compliant configuration converter SHOULD parse all of these fields and set the corresponding fields in the generated runtime configuration:

| Image Field         | Runtime Field    |
| ------------------- | ---------------  |
| `Config.User`       | `process.user.*` |

The method of parsing the above image fields are described in the following sections.

### `Config.User`

If the values of [`user` or `group`](config.md#properties) in `Config.User` are numeric (`uid` or `gid`) then the values MUST be copied verbatim to `process.user.uid` and `process.user.gid` respectively.
If the values of [`user` or `group`](config.md#properties) in `Config.User` are not numeric (`user` or `group`) then a converter SHOULD resolve the user information using a method appropriate for the container's context.
For Unix-like systems, this MAY involve resolution through NSS or parsing `/etc/passwd` from the extracted container's root filesystem to determine the values of `process.user.uid` and `process.user.gid`.

In addition, a converter SHOULD set the value of `process.user.additionalGids` to a value corresponding to the user in the container's context described by `Config.User`.
For Unix-like systems, this MAY involve resolution through NSS or parsing `/etc/group` and determining the group memberships of the user specified in `process.user.uid`.
If the value of [`user`](config.md#properties) in `Config.User` is numeric, the converter SHOULD NOT modify `process.user.additionalGids`.

If `Config.User` is not defined, the converted `process.user` value is implementation-defined.
If `Config.User` does not correspond to a user in the container's context, the converter MUST return an error.

## Optional Fields

Certain image configuration fields are not applicable to all conversion use cases, and thus are optional for configuration converters to implement.
A compliant configuration converter SHOULD provide a way for users to extract these fields into the generated runtime configuration:

| Image Field           | Runtime Field      | Notes |
| --------------------- | ------------------ | ----- |
| `Config.ExposedPorts` | `annotations`      | 1     |
| `Config.Volumes`      | `mounts`           | 2     |

1. The runtime configuration does not have a corresponding field for this image field.
   However, converters SHOULD set the [`org.opencontainers.image.exposedPorts` annotation](#config.exposedports).
2. If a converter implements conversion for this field using mountpoints, it SHOULD set the `destination` of the mountpoint to the value specified in `Config.Volumes`.
   The other `mounts` fields are platform and context dependent, and thus are implementation-defined.
   Note that the implementation of `Config.Volumes` need not use mountpoints, as it is effectively a mask of the filesystem.

### `Config.ExposedPorts`

The OCI runtime configuration does not provide a way of expressing the concept of "container exposed ports".
However, converters SHOULD set the **org.opencontainers.image.exposedPorts** annotation, unless doing so will [cause a conflict](#annotations).

**org.opencontainers.image.exposedPorts** is the list of values that correspond to the [keys defined for `Config.ExposedPorts`](config.md) (string, comma-separated values).

## Annotations

There are three ways of annotating an OCI image in this specification:

1. `Config.Labels` in the [configuration](config.md) of the image.
2. `annotations` in the [manifest](manifest.md) of the image.
3. `annotations` in the [image index](image-index.md) of the image.

In addition, there are also implicit annotations that are defined by this section which are determined from the values of the image configuration.
A converter SHOULD NOT attempt to extract annotations from [manifests](manifest.md) or [image indices](image-index.md).
If there is a conflict (same key but different value) between an implicit annotation (or annotation in [manifests](manifest.md) or [image indices](image-index.md)) and an explicitly specified annotation in `Config.Labels`, the value specified in `Config.Labels` MUST take precedence.

A converter MAY add annotations which have keys not specified in the image.
A converter MUST NOT modify the values of annotations specified in the image.
