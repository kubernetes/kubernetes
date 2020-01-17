/*
Copyright 2019 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/runtime"

	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
)

const (
	policyTemplate = `
apiVersion: %s
kind: Policy
extenders:
- urlPrefix: http://localhost:8888/
  filterVerb: filter
  prioritizeVerb: prioritize
  weight: 1
  enableHttps: false
`
)

func TestSchedulerPolicy(t *testing.T) {
	expected := &config.Policy{
		Extenders: []config.Extender{
			{
				URLPrefix:      "http://localhost:8888/",
				FilterVerb:     "filter",
				PrioritizeVerb: "prioritize",
				Weight:         1,
				EnableHTTPS:    false,
			},
		},
	}
	testcases := []struct {
		name        string
		apiVersion  string
		expectError bool
		expectedObj *config.Policy
	}{
		// verifies if a Policy YAML with apiVersion 'v1' can be
		// serialized into an unversioned Policy object.
		{
			name:        "legacy v1",
			apiVersion:  "v1",
			expectError: false,
			expectedObj: expected,
		},
		// verifies if a Policy YAML with apiVersion 'kubescheduler.config.k8s.io/v1'
		// can be serialized into an unversioned Policy object.
		{
			name:        "v1",
			apiVersion:  "kubescheduler.config.k8s.io/v1",
			expectError: false,
			expectedObj: expected,
		},
		// ensures unknown version throws a parsing error.
		{
			name:        "unknown version",
			apiVersion:  "kubescheduler.config.k8s.io/vunknown",
			expectError: true,
		},
	}
	for _, tt := range testcases {
		t.Run(tt.name, func(t *testing.T) {
			policyStr := fmt.Sprintf(policyTemplate, tt.apiVersion)
			got, err := loadPolicy([]byte(policyStr))
			if (err != nil) != tt.expectError {
				t.Fatalf("Error while parsing Policy. expectErr=%v, but got=%v.", tt.expectError, err)
			}

			if !tt.expectError {
				if diff := cmp.Diff(tt.expectedObj, got); diff != "" {
					t.Errorf("Unexpected policy diff (-want, +got): %s", diff)
				}
			}
		})
	}
}

// loadPolicy decodes data as a Policy object.
func loadPolicy(data []byte) (*config.Policy, error) {
	policy := config.Policy{}
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), data, &policy); err != nil {
		return nil, err
	}
	return &policy, nil
}
