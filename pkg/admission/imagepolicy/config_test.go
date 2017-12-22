/*
Copyright 2016 The Kubernetes Authors.

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

package imagepolicy

import (
	"reflect"
	"testing"
	"time"
)

func TestConfigNormalization(t *testing.T) {
	tests := []struct {
		test             string
		config           imagePolicyWebhookConfig
		normalizedConfig imagePolicyWebhookConfig
		wantErr          bool
	}{
		{
			test: "config within normal ranges",
			config: imagePolicyWebhookConfig{
				AllowTTL:     ((minAllowTTL + maxAllowTTL) / 2) / time.Second,
				DenyTTL:      ((minDenyTTL + maxDenyTTL) / 2) / time.Second,
				RetryBackoff: ((minRetryBackoff + maxRetryBackoff) / 2) / time.Millisecond,
			},
			normalizedConfig: imagePolicyWebhookConfig{
				AllowTTL:     ((minAllowTTL + maxAllowTTL) / 2) / time.Second * time.Second,
				DenyTTL:      ((minDenyTTL + maxDenyTTL) / 2) / time.Second * time.Second,
				RetryBackoff: (minRetryBackoff + maxRetryBackoff) / 2,
			},
			wantErr: false,
		},
		{
			test: "config below normal ranges, error",
			config: imagePolicyWebhookConfig{
				AllowTTL:     minAllowTTL - time.Duration(1),
				DenyTTL:      minDenyTTL - time.Duration(1),
				RetryBackoff: minRetryBackoff - time.Duration(1),
			},
			wantErr: true,
		},
		{
			test: "config above normal ranges, error",
			config: imagePolicyWebhookConfig{
				AllowTTL:     time.Duration(1) + maxAllowTTL,
				DenyTTL:      time.Duration(1) + maxDenyTTL,
				RetryBackoff: time.Duration(1) + maxRetryBackoff,
			},
			wantErr: true,
		},
		{
			test: "config wants default values",
			config: imagePolicyWebhookConfig{
				AllowTTL:     useDefault,
				DenyTTL:      useDefault,
				RetryBackoff: useDefault,
			},
			normalizedConfig: imagePolicyWebhookConfig{
				AllowTTL:     defaultAllowTTL,
				DenyTTL:      defaultDenyTTL,
				RetryBackoff: defaultRetryBackoff,
			},
			wantErr: false,
		},
		{
			test: "config wants disabled values",
			config: imagePolicyWebhookConfig{
				AllowTTL:     disableTTL,
				DenyTTL:      disableTTL,
				RetryBackoff: disableTTL,
			},
			normalizedConfig: imagePolicyWebhookConfig{
				AllowTTL:     time.Duration(0),
				DenyTTL:      time.Duration(0),
				RetryBackoff: time.Duration(0),
			},
			wantErr: false,
		},
		{
			test: "config within normal ranges for min values",
			config: imagePolicyWebhookConfig{
				AllowTTL:     minAllowTTL / time.Second,
				DenyTTL:      minDenyTTL / time.Second,
				RetryBackoff: minRetryBackoff,
			},
			normalizedConfig: imagePolicyWebhookConfig{
				AllowTTL:     minAllowTTL,
				DenyTTL:      minDenyTTL,
				RetryBackoff: minRetryBackoff * time.Millisecond,
			},
			wantErr: false,
		},
		{
			test: "config within normal ranges for max values",
			config: imagePolicyWebhookConfig{
				AllowTTL:     maxAllowTTL / time.Second,
				DenyTTL:      maxDenyTTL / time.Second,
				RetryBackoff: maxRetryBackoff / time.Millisecond,
			},
			normalizedConfig: imagePolicyWebhookConfig{
				AllowTTL:     maxAllowTTL,
				DenyTTL:      maxDenyTTL,
				RetryBackoff: maxRetryBackoff,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		err := normalizeWebhookConfig(&tt.config)
		if err == nil && tt.wantErr == true {
			t.Errorf("%s: expected error from normalization and didn't have one", tt.test)
		}
		if err != nil && tt.wantErr == false {
			t.Errorf("%s: unexpected error from normalization: %v", tt.test, err)
		}
		if err == nil && !reflect.DeepEqual(tt.config, tt.normalizedConfig) {
			t.Errorf("%s: expected config to be normalized. got: %v expected: %v", tt.test, tt.config, tt.normalizedConfig)
		}
	}
}
