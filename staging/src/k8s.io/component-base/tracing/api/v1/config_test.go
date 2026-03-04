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

package v1

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateTracingConfiguration(t *testing.T) {
	samplingRate := int32(12378)
	negativeRate := int32(-1)
	tooHighRate := int32(1000001)
	validEndpoint := "localhost:4317"
	dnsEndpoint := "dns://google.com:4317"
	unixEndpoint := "unix://path/to/socket"
	invalidURL := "dn%2s://localhost:4317"
	httpEndpoint := "http://localhost:4317"
	testcases := []struct {
		name        string
		expectError bool
		contents    *TracingConfiguration
	}{
		{
			name:        "sampling-rate-valid",
			expectError: false,
			contents: &TracingConfiguration{
				SamplingRatePerMillion: &samplingRate,
			},
		},
		{
			name:        "sampling-rate-negative",
			expectError: true,
			contents: &TracingConfiguration{
				SamplingRatePerMillion: &negativeRate,
			},
		},
		{
			name:        "sampling-rate-negative",
			expectError: true,
			contents: &TracingConfiguration{
				SamplingRatePerMillion: &tooHighRate,
			},
		},
		{
			name:        "default Endpoint",
			expectError: false,
			contents: &TracingConfiguration{
				Endpoint: &validEndpoint,
			},
		},
		{
			name:        "dns Endpoint",
			expectError: false,
			contents: &TracingConfiguration{
				Endpoint: &dnsEndpoint,
			},
		},
		{
			name:        "unix Endpoint",
			expectError: false,
			contents: &TracingConfiguration{
				Endpoint: &unixEndpoint,
			},
		},
		{
			name:        "invalid Endpoint",
			expectError: true,
			contents: &TracingConfiguration{
				Endpoint: &httpEndpoint,
			},
		},
		{
			name:        "invalid url",
			expectError: true,
			contents: &TracingConfiguration{
				Endpoint: &invalidURL,
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			errs := ValidateTracingConfiguration(tc.contents, nil, field.NewPath("tracing"))
			if !tc.expectError && len(errs) != 0 {
				t.Errorf("Calling ValidateTracingConfiguration expected no error, got %v", errs)
			} else if tc.expectError && len(errs) == 0 {
				t.Errorf("Calling ValidateTracingConfiguration expected error, got no error")
			}
		})
	}
}
