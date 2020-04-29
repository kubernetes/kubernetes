// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package correlation

import (
	"context"
	"net/url"
	"strings"

	"go.opentelemetry.io/otel/api/core"
	"go.opentelemetry.io/otel/api/key"
	"go.opentelemetry.io/otel/api/propagation"
)

const correlationContextHeader = "Correlation-Context"

// CorrelationContext propagates Key:Values in W3C CorrelationContext
// format.
// nolint:golint
type CorrelationContext struct{}

var _ propagation.HTTPPropagator = CorrelationContext{}

// DefaultHTTPPropagator returns the default context correlation HTTP
// propagator.
func DefaultHTTPPropagator() propagation.HTTPPropagator {
	return CorrelationContext{}
}

// Inject implements HTTPInjector.
func (CorrelationContext) Inject(ctx context.Context, supplier propagation.HTTPSupplier) {
	correlationCtx := MapFromContext(ctx)
	firstIter := true
	var headerValueBuilder strings.Builder
	correlationCtx.Foreach(func(kv core.KeyValue) bool {
		if !firstIter {
			headerValueBuilder.WriteRune(',')
		}
		firstIter = false
		headerValueBuilder.WriteString(url.QueryEscape(strings.TrimSpace((string)(kv.Key))))
		headerValueBuilder.WriteRune('=')
		headerValueBuilder.WriteString(url.QueryEscape(strings.TrimSpace(kv.Value.Emit())))
		return true
	})
	if headerValueBuilder.Len() > 0 {
		headerString := headerValueBuilder.String()
		supplier.Set(correlationContextHeader, headerString)
	}
}

// Extract implements HTTPExtractor.
func (CorrelationContext) Extract(ctx context.Context, supplier propagation.HTTPSupplier) context.Context {
	correlationContext := supplier.Get(correlationContextHeader)
	if correlationContext == "" {
		return ContextWithMap(ctx, NewEmptyMap())
	}

	contextValues := strings.Split(correlationContext, ",")
	keyValues := make([]core.KeyValue, 0, len(contextValues))
	for _, contextValue := range contextValues {
		valueAndProps := strings.Split(contextValue, ";")
		if len(valueAndProps) < 1 {
			continue
		}
		nameValue := strings.Split(valueAndProps[0], "=")
		if len(nameValue) < 2 {
			continue
		}
		name, err := url.QueryUnescape(nameValue[0])
		if err != nil {
			continue
		}
		trimmedName := strings.TrimSpace(name)
		value, err := url.QueryUnescape(nameValue[1])
		if err != nil {
			continue
		}
		trimmedValue := strings.TrimSpace(value)

		// TODO (skaris): properties defiend https://w3c.github.io/correlation-context/, are currently
		// just put as part of the value.
		var trimmedValueWithProps strings.Builder
		trimmedValueWithProps.WriteString(trimmedValue)
		for _, prop := range valueAndProps[1:] {
			trimmedValueWithProps.WriteRune(';')
			trimmedValueWithProps.WriteString(prop)
		}

		keyValues = append(keyValues, key.New(trimmedName).String(trimmedValueWithProps.String()))
	}
	return ContextWithMap(ctx, NewMap(MapUpdate{
		MultiKV: keyValues,
	}))
}

// GetAllKeys implements HTTPPropagator.
func (CorrelationContext) GetAllKeys() []string {
	return []string{correlationContextHeader}
}
