// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

/*
Package log provides the OpenTelemetry Logs API.

This API is separate from its implementation, so the instrumentation built
from it is reusable. See [go.opentelemetry.io/otel/sdk/log] for the official
OpenTelemetry implementation of this API.

The log package provides the OpenTelemetry Logs API, which serves as a standard
interface for generating and managing log records within the OpenTelemetry ecosystem.
This package allows users to emit LogRecords, enabling structured, context-rich logging
that can be easily integrated with observability tools. It ensures that log data is captured
in a way that is consistent with OpenTelemetry's data model.

This package can be used to create bridges between existing logging libraries and OpenTelemetry.
Log bridges allow existing logging setups to be integrated with OpenTelemetry.
Log bridges can be found in the [registry].

# API Implementations

This package does not conform to the standard Go versioning policy; all of its
interfaces may have methods added to them without a package major version bump.
This non-standard API evolution could surprise an uninformed implementation
author. They could unknowingly build their implementation in a way that would
result in a runtime panic for their users who update to the new API.

The API is designed to help inform an instrumentation author about this
non-standard API evolution. It requires them to choose a default behavior for
unimplemented interface methods. There are three behavior choices they can
make:

  - Compilation failure
  - Panic
  - Default to another implementation

All interfaces in this API embed a corresponding interface from
[go.opentelemetry.io/otel/log/embedded]. If an author wants the default
behavior of their implementations to be a compilation failure, signaling to
their users that they need to update to the latest version of that implementation,
they need to embed the corresponding interface from
[go.opentelemetry.io/otel/log/embedded] in their implementation. For example,

	import "go.opentelemetry.io/otel/log/embedded"

	type LoggerProvider struct {
		embedded.LoggerProvider
		// ...
	}

If an author wants the default behavior of their implementations to panic,
they need to embed the API interface directly.

	import "go.opentelemetry.io/otel/log"

	type LoggerProvider struct {
		log.LoggerProvider
		// ...
	}

This is not a recommended behavior as it could lead to publishing packages that
contain runtime panics when users update other packages that use newer versions
of [go.opentelemetry.io/otel/log].

Finally, an author can embed another implementation in theirs. The embedded
implementation will be used for methods not defined by the author. For example,
an author who wants to default to silently dropping the call can use
[go.opentelemetry.io/otel/log/noop]:

	import "go.opentelemetry.io/otel/log/noop"

	type LoggerProvider struct {
		noop.LoggerProvider
		// ...
	}

It is strongly recommended that authors only embed
go.opentelemetry.io/otel/log/noop if they choose this default behavior. That
implementation is the only one OpenTelemetry authors can guarantee will fully
implement all the API interfaces when a user updates their API.

[registry]: https://opentelemetry.io/ecosystem/registry/?language=go&component=log-bridge
*/
package log
