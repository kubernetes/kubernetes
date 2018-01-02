# <a name="openContainerInitiativeRuntimeSpecification" />Open Container Initiative Runtime Specification

The [Open Container Initiative][oci] develops specifications for standards on Operating System process and application containers.

# <a name="ociRuntimeSpecAbstract" />Abstract

The OCI Runtime Specification aims to specify the configuration, execution environment, and lifecycle of a container.

A container's configuration is specified as the `config.json` for the supported platforms and details the fields that enable the creation of a container.
The execution environment is specified to ensure that applications running inside a container have a consistent environment between runtimes along with common actions defined for the container's lifecycle.

# <a name="ociRuntimeSpecPlatforms" />Platforms

Platforms defined by this specification are:

* `linux`: [runtime.md](runtime.md), [config.md](config.md), [config-linux.md](config-linux.md), and [runtime-linux.md](runtime-linux.md).
* `solaris`: [runtime.md](runtime.md), [config.md](config.md), and [config-solaris.md](config-solaris.md).
* `windows`: [runtime.md](runtime.md), [config.md](config.md), and [config-windows.md](config-windows.md).

# <a name="ociRuntimeSpecTOC" />Table of Contents

- [Introduction](spec.md)
    - [Notational Conventions](#notational-conventions)
    - [Container Principles](principles.md)
- [Filesystem Bundle](bundle.md)
- [Runtime and Lifecycle](runtime.md)
    - [Linux-specific Runtime and Lifecycle](runtime-linux.md)
- [Configuration](config.md)
    - [Linux-specific Configuration](config-linux.md)
    - [Solaris-specific Configuration](config-solaris.md)
    - [Windows-specific Configuration](config-windows.md)
- [Glossary](glossary.md)

# <a name="ociRuntimeSpecNotationalConventions" />Notational Conventions

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" are to be interpreted as described in [RFC 2119][rfc2119].

The key words "unspecified", "undefined", and "implementation-defined" are to be interpreted as described in the [rationale for the C99 standard][c99-unspecified].

An implementation is not compliant for a given CPU architecture if it fails to satisfy one or more of the MUST, REQUIRED, or SHALL requirements for the [platforms](#platforms) it implements.
An implementation is compliant for a given CPU architecture if it satisfies all the MUST, REQUIRED, and SHALL requirements for the [platforms](#platforms) it implements.


[c99-unspecified]: http://www.open-std.org/jtc1/sc22/wg14/www/C99RationaleV5.10.pdf#page=18
[oci]: http://www.opencontainers.org
[rfc2119]: http://tools.ietf.org/html/rfc2119
