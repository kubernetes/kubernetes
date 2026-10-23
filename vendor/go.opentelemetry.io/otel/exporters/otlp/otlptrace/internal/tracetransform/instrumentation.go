// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package tracetransform

import (
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"

	"go.opentelemetry.io/otel/sdk/instrumentation"
)

// InstrumentationScope transforms il into an OTLP InstrumentationScope,
// returning nil if il is the zero value.
func InstrumentationScope(il instrumentation.Scope) *commonpb.InstrumentationScope {
	if il == (instrumentation.Scope{}) {
		return nil
	}
	return &commonpb.InstrumentationScope{
		Name:       il.Name,
		Version:    il.Version,
		Attributes: Iterator(il.Attributes.Iter()),
	}
}
