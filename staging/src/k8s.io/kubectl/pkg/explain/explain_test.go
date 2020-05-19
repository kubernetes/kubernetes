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

package explain

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta/testrestmapper"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubectl/pkg/scheme"
)

func TestSplitAndParseResourceRequest(t *testing.T) {
	tests := []struct {
		name       string
		inResource string

		expectedGVR        schema.GroupVersionResource
		expectedFieldsPath []string
		expectedErr        bool
	}{
		{
			name:       "no trailing period",
			inResource: "pods.field2.field3",

			expectedGVR:        schema.GroupVersionResource{Resource: "pods", Version: "v1"},
			expectedFieldsPath: []string{"field2", "field3"},
		},
		{
			name:       "trailing period with correct fieldsPath",
			inResource: "service.field2.field3.",

			expectedGVR:        schema.GroupVersionResource{Resource: "services", Version: "v1"},
			expectedFieldsPath: []string{"field2", "field3"},
		},
		{
			name:       "trailing period with incorrect fieldsPath",
			inResource: "node.field2.field3.",

			expectedGVR:        schema.GroupVersionResource{Resource: "nodes", Version: "v1"},
			expectedFieldsPath: []string{"field2", "field3", ""},
			expectedErr:        true,
		},
	}

	mapper := testrestmapper.TestOnlyStaticRESTMapper(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotGVR, gotFieldsPath, err := SplitAndParseResourceRequest(tt.inResource, mapper)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if !reflect.DeepEqual(tt.expectedGVR, gotGVR) && !tt.expectedErr {
				t.Errorf("%s: expected inResource: %s, got: %s", tt.name, tt.expectedGVR, gotGVR)
			}

			if !reflect.DeepEqual(tt.expectedFieldsPath, gotFieldsPath) && !tt.expectedErr {
				t.Errorf("%s: expected fieldsPath: %s, got: %s", tt.name, tt.expectedFieldsPath, gotFieldsPath)
			}
		})
	}
}
