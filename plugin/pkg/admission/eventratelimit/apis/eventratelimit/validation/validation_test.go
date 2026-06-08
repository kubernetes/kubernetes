/*
Copyright 2017 The Kubernetes Authors.

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

	eventratelimitapi "k8s.io/kubernetes/plugin/pkg/admission/eventratelimit/apis/eventratelimit"
)

func TestValidateConfiguration(t *testing.T) {
	cases := []struct {
		name           string
		config         eventratelimitapi.Configuration
		expectedResult bool
	}{{
		name: "valid server",
		config: eventratelimitapi.Configuration{
			Limits: []eventratelimitapi.Limit{{
				Type:  "Server",
				Burst: 5,
				QPS:   1,
			}},
		},
		expectedResult: true,
	}, {
		name: "valid namespace",
		config: eventratelimitapi.Configuration{
			Limits: []eventratelimitapi.Limit{{
				Type:      "Namespace",
				Burst:     10,
				QPS:       2,
				CacheSize: 100,
			}},
		},
		expectedResult: true,
	}, {
		name: "valid user",
		config: eventratelimitapi.Configuration{
			Limits: []eventratelimitapi.Limit{{
				Type:      "User",
				Burst:     10,
				QPS:       2,
				CacheSize: 100,
			}},
		},
		expectedResult: true,
	}, {
		name: "valid source+object",
		config: eventratelimitapi.Configuration{
			Limits: []eventratelimitapi.Limit{{
				Type:      "SourceAndObject",
				Burst:     5,
				QPS:       1,
				CacheSize: 1000,
			}},
		},
		expectedResult: true,
	}, {
		name: "valid multiple",
		config: eventratelimitapi.Configuration{
			Limits: []eventratelimitapi.Limit{{
				Type:  "Server",
				Burst: 5,
				QPS:   1,
			}, {
				Type:      "Namespace",
				Burst:     10,
				QPS:       2,
				CacheSize: 100,
			}, {
				Type:      "SourceAndObject",
				Burst:     25,
				QPS:       10,
				CacheSize: 1000,
			}},
		},
		expectedResult: true,
	}, {
		name:           "missing limits",
		config:         eventratelimitapi.Configuration{},
		expectedResult: false,
	}, {
		name: "missing type",
		config: eventratelimitapi.Configuration{
			Limits: []eventratelimitapi.Limit{{
				Burst:     25,
				QPS:       10,
				CacheSize: 1000,
			}},
		},
		expectedResult: false,
	}, {
		name: "invalid type",
		config: eventratelimitapi.Configuration{
			Limits: []eventratelimitapi.Limit{{
				Type:      "unknown-type",
				Burst:     25,
				QPS:       10,
				CacheSize: 1000,
			}},
		},
		expectedResult: false,
	}, {
		name: "missing burst",
		config: eventratelimitapi.Configuration{
			Limits: []eventratelimitapi.Limit{{
				Type: "Server",
				QPS:  1,
			}},
		},
		expectedResult: false,
	}, {
		name: "missing qps",
		config: eventratelimitapi.Configuration{
			Limits: []eventratelimitapi.Limit{{
				Type:  "Server",
				Burst: 5,
			}},
		},
		expectedResult: false,
	}, {
		name: "negative cache size",
		config: eventratelimitapi.Configuration{
			Limits: []eventratelimitapi.Limit{{
				Type:      "Namespace",
				Burst:     10,
				QPS:       2,
				CacheSize: -1,
			}},
		},
		expectedResult: false,
	}}
	for _, tc := range cases {
		errs := ValidateConfiguration(&tc.config)
		if e, a := tc.expectedResult, len(errs) == 0; e != a {
			if e {
				t.Errorf("%v: expected success: %v", tc.name, errs)
			} else {
				t.Errorf("%v: expected failure", tc.name)
			}
		}
	}
}
