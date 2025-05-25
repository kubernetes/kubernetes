/*
Copyright 2025 The Kubernetes Authors.

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

package validation

import (
	"testing"

	internalapi "k8s.io/kubernetes/plugin/pkg/admission/defaulttolerationseconds/apis/defaulttolerationseconds"
)

func int64Ptr(i int64) *int64 { return &i }

func TestValidateConfiguration(t *testing.T) {
	tests := []struct {
		name    string
		config  *internalapi.Configuration
		wantErr bool
	}{
		{
			name:    "nil config",
			config:  nil,
			wantErr: true,
		},
		{
			name: "both zeros",
			config: &internalapi.Configuration{
				DefaultTolerationSecondsConfig: internalapi.DefaultTolerationSecondsConfig{
					NotReadyTolerationSeconds:    int64Ptr(0),
					UnreachableTolerationSeconds: int64Ptr(0),
				},
			},
			wantErr: false,
		},
		{
			name: "both positive",
			config: &internalapi.Configuration{
				DefaultTolerationSecondsConfig: internalapi.DefaultTolerationSecondsConfig{
					NotReadyTolerationSeconds:    int64Ptr(30),
					UnreachableTolerationSeconds: int64Ptr(60),
				},
			},
			wantErr: false,
		},
		{
			name: "negative NotReady",
			config: &internalapi.Configuration{
				DefaultTolerationSecondsConfig: internalapi.DefaultTolerationSecondsConfig{
					NotReadyTolerationSeconds:    int64Ptr(-1),
					UnreachableTolerationSeconds: int64Ptr(10),
				},
			},
			wantErr: true,
		},
		{
			name: "negative Unreachable",
			config: &internalapi.Configuration{
				DefaultTolerationSecondsConfig: internalapi.DefaultTolerationSecondsConfig{
					NotReadyTolerationSeconds:    int64Ptr(10),
					UnreachableTolerationSeconds: int64Ptr(-5),
				},
			},
			wantErr: true,
		},
		{
			name: "both negative",
			config: &internalapi.Configuration{
				DefaultTolerationSecondsConfig: internalapi.DefaultTolerationSecondsConfig{
					NotReadyTolerationSeconds:    int64Ptr(-1),
					UnreachableTolerationSeconds: int64Ptr(-5),
				},
			},
			wantErr: true,
		},
		{
			name: "nil pointers",
			config: &internalapi.Configuration{
				DefaultTolerationSecondsConfig: internalapi.DefaultTolerationSecondsConfig{
					NotReadyTolerationSeconds:    nil,
					UnreachableTolerationSeconds: nil,
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		err := ValidateConfiguration(tt.config)
		if (err != nil) != tt.wantErr {
			t.Errorf("Test %q: expected error=%v, got error=%v", tt.name, tt.wantErr, err != nil)
		}
	}
}
