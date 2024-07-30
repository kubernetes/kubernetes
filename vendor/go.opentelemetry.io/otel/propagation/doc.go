// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

/*
Package propagation contains OpenTelemetry context propagators.

OpenTelemetry propagators are used to extract and inject context data from and
into messages exchanged by applications. The propagator supported by this
package is the W3C Trace Context encoding
(https://www.w3.org/TR/trace-context/), and W3C Baggage
(https://www.w3.org/TR/baggage/).
*/
package propagation // import "go.opentelemetry.io/otel/propagation"
