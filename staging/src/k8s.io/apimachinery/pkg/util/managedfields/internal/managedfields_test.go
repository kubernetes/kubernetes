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
	"fmt"
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"sigs.k8s.io/yaml"
)

// TestHasFieldsType makes sure that we fail if we don't have a
// FieldsType set properly.
func TestHasFieldsType(t *testing.T) {
	var unmarshaled []metav1.ManagedFieldsEntry
	if err := yaml.Unmarshal([]byte(`- apiVersion: v1
  fieldsType: FieldsV1
  fieldsV1:
    f:field: {}
  manager: foo
  operation: Apply
`), &unmarshaled); err != nil {
		t.Fatalf("did not expect yaml unmarshalling error but got: %v", err)
	}
	if _, err := DecodeManagedFields(unmarshaled); err != nil {
		t.Fatalf("did not expect decoding error but got: %v", err)
	}

	// Invalid fieldsType V2.
	if err := yaml.Unmarshal([]byte(`- apiVersion: v1
  fieldsType: FieldsV2
  fieldsV1:
    f:field: {}
  manager: foo
  operation: Apply
`), &unmarshaled); err != nil {
		t.Fatalf("did not expect yaml unmarshalling error but got: %v", err)
	}
	if _, err := DecodeManagedFields(unmarshaled); err == nil {
		t.Fatal("Expect decoding error but got none")
	}

	// Missing fieldsType.
	if err := yaml.Unmarshal([]byte(`- apiVersion: v1
  fieldsV1:
    f:field: {}
  manager: foo
  operation: Apply
`), &unmarshaled); err != nil {
		t.Fatalf("did not expect yaml unmarshalling error but got: %v", err)
	}
	if _, err := DecodeManagedFields(unmarshaled); err == nil {
		t.Fatal("Expect decoding error but got none")
	}
}

// TestHasAPIVersion makes sure that we fail if we don't have an
// APIVersion set.
func TestHasAPIVersion(t *testing.T) {
	var unmarshaled []metav1.ManagedFieldsEntry
	if err := yaml.Unmarshal([]byte(`- apiVersion: v1
  fieldsType: FieldsV1
  fieldsV1:
    f:field: {}
  manager: foo
  operation: Apply
`), &unmarshaled); err != nil {
		t.Fatalf("did not expect yaml unmarshalling error but got: %v", err)
	}
	if _, err := DecodeManagedFields(unmarshaled); err != nil {
		t.Fatalf("did not expect decoding error but got: %v", err)
	}

	// Missing apiVersion.
	unmarshaled = nil
	if err := yaml.Unmarshal([]byte(`- fieldsType: FieldsV1
  fieldsV1:
    f:field: {}
  manager: foo
  operation: Apply
`), &unmarshaled); err != nil {
		t.Fatalf("did not expect yaml unmarshalling error but got: %v", err)
	}
	if _, err := DecodeManagedFields(unmarshaled); err == nil {
		t.Fatal("Expect decoding error but got none")
	}
}

// TestHasOperation makes sure that we fail if we don't have an
// Operation set properly.
func TestHasOperation(t *testing.T) {
	var unmarshaled []metav1.ManagedFieldsEntry
	if err := yaml.Unmarshal([]byte(`- apiVersion: v1
  fieldsType: FieldsV1
  fieldsV1:
    f:field: {}
  manager: foo
  operation: Apply
`), &unmarshaled); err != nil {
		t.Fatalf("did not expect yaml unmarshalling error but got: %v", err)
	}
	if _, err := DecodeManagedFields(unmarshaled); err != nil {
		t.Fatalf("did not expect decoding error but got: %v", err)
	}

	// Invalid operation.
	if err := yaml.Unmarshal([]byte(`- apiVersion: v1
  fieldsType: FieldsV1
  fieldsV1:
    f:field: {}
  manager: foo
  operation: Invalid
`), &unmarshaled); err != nil {
		t.Fatalf("did not expect yaml unmarshalling error but got: %v", err)
	}
	if _, err := DecodeManagedFields(unmarshaled); err == nil {
		t.Fatal("Expect decoding error but got none")
	}

	// Missing operation.
	unmarshaled = nil
	if err := yaml.Unmarshal([]byte(`- apiVersion: v1
  fieldsType: FieldsV1
  fieldsV1:
    f:field: {}
  manager: foo
`), &unmarshaled); err != nil {
		t.Fatalf("did not expect yaml unmarshalling error but got: %v", err)
	}
	if _, err := DecodeManagedFields(unmarshaled); err == nil {
		t.Fatal("Expect decoding error but got none")
	}
}

// TestRoundTripManagedFields will roundtrip ManagedFields from the wire format
// (api format) to the format used by sigs.k8s.io/structured-merge-diff and back
func TestRoundTripManagedFields(t *testing.T) {
	tests := []string{
		`null
`,
		`- apiVersion: v1
  fieldsType: FieldsV1
  fieldsV1:
    v:3:
      f:alsoPi: {}
    v:3.1415:
      f:pi: {}
    v:false:
      f:notTrue: {}
  manager: foo
  operation: Update
  time: "2001-02-03T04:05:06Z"
- apiVersion: v1beta1
  fieldsType: FieldsV1
  fieldsV1:
    i:5:
      f:i: {}
  manager: foo
  operation: Update
  time: "2011-12-13T14:15:16Z"
`,
		`- apiVersion: v1
  fieldsType: FieldsV1
  fieldsV1:
    f:spec:
      f:containers:
        k:{"name":"c"}:
          f:image: {}
          f:name: {}
  manager: foo
  operation: Apply
`,
		`- apiVersion: v1
  fieldsType: FieldsV1
  fieldsV1:
    f:apiVersion: {}
    f:kind: {}
    f:metadata:
      f:labels:
        f:app: {}
      f:name: {}
    f:spec:
      f:replicas: {}
      f:selector:
        f:matchLabels:
          f:app: {}
      f:template:
        f:medatada:
          f:labels:
            f:app: {}
        f:spec:
          f:containers:
            k:{"name":"nginx"}:
              .: {}
              f:image: {}
              f:name: {}
              f:ports:
                i:0:
                  f:containerPort: {}
  manager: foo
  operation: Update
`,
		`- apiVersion: v1
  fieldsType: FieldsV1
  fieldsV1:
    f:allowVolumeExpansion: {}
    f:apiVersion: {}
    f:kind: {}
    f:metadata:
      f:name: {}
      f:parameters:
        f:resturl: {}
        f:restuser: {}
        f:secretName: {}
        f:secretNamespace: {}
    f:provisioner: {}
  manager: foo
  operation: Apply
`,
		`- apiVersion: v1
  fieldsType: FieldsV1
  fieldsV1:
    f:apiVersion: {}
    f:kind: {}
    f:metadata:
      f:name: {}
    f:spec:
      f:group: {}
      f:names:
        f:kind: {}
        f:plural: {}
        f:shortNames:
          i:0: {}
        f:singular: {}
      f:scope: {}
      f:versions:
        k:{"name":"v1"}:
          f:name: {}
          f:served: {}
          f:storage: {}
  manager: foo
  operation: Update
`,
		`- apiVersion: v1
  fieldsType: FieldsV1
  fieldsV1:
    f:spec:
      f:replicas: {}
  manager: foo
  operation: Update
  subresource: scale
`,
	}

	for _, test := range tests {
		t.Run(test, func(t *testing.T) {
			var unmarshaled []metav1.ManagedFieldsEntry
			if err := yaml.Unmarshal([]byte(test), &unmarshaled); err != nil {
				t.Fatalf("did not expect yaml unmarshalling error but got: %v", err)
			}
			decoded, err := DecodeManagedFields(unmarshaled)
			if err != nil {
				t.Fatalf("did not expect decoding error but got: %v", err)
			}
			encoded, err := encodeManagedFields(decoded)
			if err != nil {
				t.Fatalf("did not expect encoding error but got: %v", err)
			}
			marshaled, err := yaml.Marshal(&encoded)
			if err != nil {
				t.Fatalf("did not expect yaml marshalling error but got: %v", err)
			}
			if !reflect.DeepEqual(string(marshaled), test) {
				t.Fatalf("expected:\n%v\nbut got:\n%v", test, string(marshaled))
			}
		})
	}
}

func TestBuildManagerIdentifier(t *testing.T) {
	tests := []struct {
		managedFieldsEntry string
		expected           string
	}{
		{
			managedFieldsEntry: `
apiVersion: v1
fieldsV1:
  f:apiVersion: {}
manager: foo
operation: Update
time: "2001-02-03T04:05:06Z"
`,
			expected: "{\"manager\":\"foo\",\"operation\":\"Update\",\"apiVersion\":\"v1\"}",
		},
		{
			managedFieldsEntry: `
apiVersion: v1
fieldsV1:
  f:apiVersion: {}
manager: foo
operation: Apply
time: "2001-02-03T04:05:06Z"
`,
			expected: "{\"manager\":\"foo\",\"operation\":\"Apply\"}",
		},
		{
			managedFieldsEntry: `
apiVersion: v1
fieldsV1:
  f:apiVersion: {}
manager: foo
operation: Apply
subresource: scale
time: "2001-02-03T04:05:06Z"
`,
			expected: "{\"manager\":\"foo\",\"operation\":\"Apply\",\"subresource\":\"scale\"}",
		},
	}

	for _, test := range tests {
		t.Run(test.managedFieldsEntry, func(t *testing.T) {
			var unmarshaled metav1.ManagedFieldsEntry
			if err := yaml.Unmarshal([]byte(test.managedFieldsEntry), &unmarshaled); err != nil {
				t.Fatalf("did not expect yaml unmarshalling error but got: %v", err)
			}
			decoded, err := BuildManagerIdentifier(&unmarshaled)
			if err != nil {
				t.Fatalf("did not expect decoding error but got: %v", err)
			}
			if !reflect.DeepEqual(decoded, test.expected) {
				t.Fatalf("expected:\n%v\nbut got:\n%v", test.expected, decoded)
			}
		})
	}
}

func TestSortEncodedManagedFields(t *testing.T) {
	tests := []struct {
		name          string
		managedFields []metav1.ManagedFieldsEntry
		expected      []metav1.ManagedFieldsEntry
	}{
		{
			name:          "empty",
			managedFields: []metav1.ManagedFieldsEntry{},
			expected:      []metav1.ManagedFieldsEntry{},
		},
		{
			name:          "nil",
			managedFields: nil,
			expected:      nil,
		},
		{
			name: "remains untouched",
			managedFields: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
			},
			expected: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
			},
		},
		{
			name: "manager without time first",
			managedFields: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
			},
			expected: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
			},
		},
		{
			name: "manager without time first name last",
			managedFields: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
				{Manager: "b", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
			},
			expected: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "b", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
			},
		},
		{
			name: "apply first",
			managedFields: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
			},
			expected: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
			},
		},
		{
			name: "newest last",
			managedFields: []metav1.ManagedFieldsEntry{
				{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2002-01-01T01:00:00Z")},
				{Manager: "b", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
			},
			expected: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "b", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
				{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2002-01-01T01:00:00Z")},
			},
		},
		{
			name: "manager last",
			managedFields: []metav1.ManagedFieldsEntry{
				{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "d", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
				{Manager: "b", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
			},
			expected: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "b", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
				{Manager: "d", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
			},
		},
		{
			name: "manager sorted",
			managedFields: []metav1.ManagedFieldsEntry{
				{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "g", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "f", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2002-01-01T01:00:00Z")},
				{Manager: "i", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "d", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2002-01-01T01:00:00Z")},
				{Manager: "h", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "e", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2003-01-01T01:00:00Z")},
				{Manager: "b", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
			},
			expected: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "b", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "g", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "h", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "i", Operation: metav1.ManagedFieldsOperationApply, Time: nil},
				{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2001-01-01T01:00:00Z")},
				{Manager: "d", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2002-01-01T01:00:00Z")},
				{Manager: "f", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2002-01-01T01:00:00Z")},
				{Manager: "e", Operation: metav1.ManagedFieldsOperationUpdate, Time: parseTimeOrPanic("2003-01-01T01:00:00Z")},
			},
		},
		{
			name: "sort drops nanoseconds",
			managedFields: []metav1.ManagedFieldsEntry{
				{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Date(2000, time.January, 0, 0, 0, 0, 1, time.UTC)}},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Date(2000, time.January, 0, 0, 0, 0, 2, time.UTC)}},
				{Manager: "b", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Date(2000, time.January, 0, 0, 0, 0, 3, time.UTC)}},
			},
			expected: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Date(2000, time.January, 0, 0, 0, 0, 2, time.UTC)}},
				{Manager: "b", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Date(2000, time.January, 0, 0, 0, 0, 3, time.UTC)}},
				{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Date(2000, time.January, 0, 0, 0, 0, 1, time.UTC)}},
			},
		},
		{
			name: "entries with subresource field",
			managedFields: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Subresource: "status"},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Subresource: "scale"},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply},
			},
			expected: []metav1.ManagedFieldsEntry{
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Subresource: "scale"},
				{Manager: "a", Operation: metav1.ManagedFieldsOperationApply, Subresource: "status"},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			sorted, err := sortEncodedManagedFields(test.managedFields)
			if err != nil {
				t.Fatalf("did not expect error when sorting but got: %v", err)
			}
			if !reflect.DeepEqual(sorted, test.expected) {
				t.Fatalf("expected:\n%v\nbut got:\n%v", test.expected, sorted)
			}
		})
	}
}

func parseTimeOrPanic(s string) *metav1.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(fmt.Sprintf("failed to parse time %s, got: %v", s, err))
	}
	return &metav1.Time{Time: t.UTC()}
}
