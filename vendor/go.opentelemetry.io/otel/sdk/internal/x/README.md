# Experimental Features

The SDK contains features that have not yet stabilized in the OpenTelemetry specification.
These features are added to the OpenTelemetry Go SDK prior to stabilization in the specification so that users can start experimenting with them and provide feedback.

These feature may change in backwards incompatible ways as feedback is applied.
See the [Compatibility and Stability](#compatibility-and-stability) section for more information.

## Features

- [Resource](#resource)

### Resource

[OpenTelemetry resource semantic conventions] include many attribute definitions that are defined as experimental.
To have experimental semantic conventions be added by [resource detectors] set the `OTEL_GO_X_RESOURCE` environment variable.
The value set must be the case-insensitive string of `"true"` to enable the feature.
All other values are ignored.

<!-- TODO: document what attributes are added by which detector -->

[OpenTelemetry resource semantic conventions]: https://opentelemetry.io/docs/specs/semconv/resource/
[resource detectors]: https://pkg.go.dev/go.opentelemetry.io/otel/sdk/resource#Detector

#### Examples

Enable experimental resource semantic conventions.

```console
export OTEL_GO_X_RESOURCE=true
```

Disable experimental resource semantic conventions.

```console
unset OTEL_GO_X_RESOURCE
```

## Compatibility and Stability

Experimental features do not fall within the scope of the OpenTelemetry Go versioning and stability [policy](../../../VERSIONING.md).
These features may be removed or modified in successive version releases, including patch versions.

When an experimental feature is promoted to a stable feature, a migration path will be included in the changelog entry of the release.
There is no guarantee that any environment variable feature flags that enabled the experimental feature will be supported by the stable version.
If they are supported, they may be accompanied with a deprecation notice stating a timeline for the removal of that support.
