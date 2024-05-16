/*
Copyright 2022 The Kubernetes Authors.

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

package v1

// TracingConfiguration provides versioned configuration for OpenTelemetry tracing clients.
type TracingConfiguration struct {
	// Endpoint of the collector this component will report traces to.
	// The connection is insecure, and does not currently support TLS.
	// Recommended is unset, and endpoint is the otlp grpc default, localhost:4317.
	// +optional
	Endpoint *string `json:"endpoint,omitempty"`

	// SamplingRatePerMillion is the number of samples to collect per million spans.
	// Recommended is unset. If unset, sampler respects its parent span's sampling
	// rate, but otherwise never samples.
	// +optional
	SamplingRatePerMillion *int32 `json:"samplingRatePerMillion,omitempty"`
}
