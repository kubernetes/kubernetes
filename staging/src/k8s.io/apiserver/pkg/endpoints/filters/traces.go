/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package filters

import (
	"net/http"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel/trace"

	"k8s.io/component-base/traces"
)

// WithTracing adds tracing to requests if the incoming request is sampled
func WithTracing(handler http.Handler, tp *trace.TracerProvider) http.Handler {
	opts := []otelhttp.Option{
		otelhttp.WithPropagators(traces.Propagators()),
		otelhttp.WithPublicEndpoint(),
	}
	if tp != nil {
		opts = append(opts, otelhttp.WithTracerProvider(*tp))
	}
	// Even if there is no TracerProvider, the otelhttp still handles context propagation.
	// See https://github.com/open-telemetry/opentelemetry-go/tree/main/example/passthrough
	return otelhttp.NewHandler(handler, "KubernetesAPI", opts...)
}
