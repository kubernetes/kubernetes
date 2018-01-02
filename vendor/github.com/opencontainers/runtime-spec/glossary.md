# <a name="glossary" />Glossary

## <a name="glossaryBundle" />Bundle

A [directory structure](bundle.md) that is written ahead of time, distributed, and used to seed the runtime for creating a [container](#container) and launching a process within it.

## <a name="glossaryConfiguration" />Configuration

The [`config.json`](config.md) file in a [bundle](#bundle) which defines the intended [container](#container) and container process.

## <a name="glossaryContainer" />Container

An environment for executing processes with configurable isolation and resource limitations.
For example, namespaces, resource limits, and mounts are all part of the container environment.

## <a name="glossaryContainerNamespace" />Container namespace

On Linux,the [namespaces][namespaces.7] in which the [configured process](config.md#process) executes.

## <a name="glossaryJson" />JSON

All configuration [JSON][] MUST be encoded in [UTF-8][].
JSON objects MUST NOT include duplicate names.
The order of entries in JSON objects is not significant.

## <a name="glossaryRuntime" />Runtime

An implementation of this specification.
It reads the [configuration files](#configuration) from a [bundle](#bundle), uses that information to create a [container](#container), launches a process inside the container, and performs other [lifecycle actions](runtime.md).

## <a name="glossaryRuntimeNamespace" />Runtime namespace

On Linux, the namespaces from which new [container namespaces](#container-namespace) are [created](config-linux.md#namespaces) and from which some configured resources are accessed.

[JSON]: https://tools.ietf.org/html/rfc7159
[UTF-8]: http://www.unicode.org/versions/Unicode8.0.0/ch03.pdf

[namespaces.7]: http://man7.org/linux/man-pages/man7/namespaces.7.html
