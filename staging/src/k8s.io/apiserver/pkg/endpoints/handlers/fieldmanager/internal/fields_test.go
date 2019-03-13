/*
Copyright 2018 The Kubernetes Authors.

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

package internal

import (
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
)

// TestFieldsRoundTrip tests that a fields trie can be round tripped as a path set
func TestFieldsRoundTrip(t *testing.T) {
	tests := []metav1.Fields{
		{
			Map: map[string]metav1.Fields{
				"f:metadata": {
					Map: map[string]metav1.Fields{
						".":      newFields(),
						"f:name": newFields(),
					},
				},
			},
		},
	}

	for _, test := range tests {
		set, err := FieldsToSet(test)
		if err != nil {
			t.Fatalf("Failed to create path set: %v", err)
		}
		output, err := SetToFields(set)
		if err != nil {
			t.Fatalf("Failed to create fields trie from path set: %v", err)
		}
		if !reflect.DeepEqual(test, output) {
			t.Fatalf("Expected round-trip:\ninput: %v\noutput: %v", test, output)
		}
	}
}

// TestFieldsToSetError tests that errors are picked up by FieldsToSet
func TestFieldsToSetError(t *testing.T) {
	tests := []struct {
		fields    metav1.Fields
		errString string
	}{
		{
			fields: metav1.Fields{
				Map: map[string]metav1.Fields{
					"k:{invalid json}": {
						Map: map[string]metav1.Fields{
							".":      newFields(),
							"f:name": newFields(),
						},
					},
				},
			},
			errString: "invalid character",
		},
	}

	for _, test := range tests {
		_, err := FieldsToSet(test.fields)
		if err == nil || !strings.Contains(err.Error(), test.errString) {
			t.Fatalf("Expected error to contain %q but got: %v", test.errString, err)
		}
	}
}

// TestSetToFieldsError tests that errors are picked up by SetToFields
func TestSetToFieldsError(t *testing.T) {
	validName := "ok"
	invalidPath := fieldpath.Path([]fieldpath.PathElement{{}, {FieldName: &validName}})

	tests := []struct {
		set       fieldpath.Set
		errString string
	}{
		{
			set:       *fieldpath.NewSet(invalidPath),
			errString: "Invalid type of path element",
		},
	}

	for _, test := range tests {
		_, err := SetToFields(test.set)
		if err == nil || !strings.Contains(err.Error(), test.errString) {
			t.Fatalf("Expected error to contain %q but got: %v", test.errString, err)
		}
	}
}
