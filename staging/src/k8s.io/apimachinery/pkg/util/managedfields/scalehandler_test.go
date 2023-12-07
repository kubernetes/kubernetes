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

package managedfields

import (
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

func TestTransformManagedFieldsToSubresource(t *testing.T) {
	testTime, _ := time.ParseInLocation("2006-Jan-02", "2013-Feb-03", time.Local)
	managedFieldTime := metav1.NewTime(testTime)

	tests := []struct {
		desc     string
		input    []metav1.ManagedFieldsEntry
		expected []metav1.ManagedFieldsEntry
	}{
		{
			desc: "filter one entry and transform it into a subresource entry",
			input: []metav1.ManagedFieldsEntry{
				{
					Manager:    "manager-1",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:another-field":{}}}`)},
				},
				{
					Manager:    "manager-2",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Time:       &managedFieldTime,
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:    "manager-2",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "autoscaling/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Time:       &managedFieldTime,
				},
			},
		},
		{
			desc: "transform all entries",
			input: []metav1.ManagedFieldsEntry{
				{
					Manager:    "manager-1",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
				},
				{
					Manager:    "manager-2",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
				},
				{
					Manager:     "manager-3",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:    "manager-1",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "autoscaling/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
				},
				{
					Manager:    "manager-2",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "autoscaling/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
				},
				{
					Manager:     "manager-3",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
		},
		{
			desc: "drops fields if the api version is unknown",
			input: []metav1.ManagedFieldsEntry{
				{
					Manager:    "manager-1",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v10",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
				},
			},
			expected: nil,
		},
	}

	for _, test := range tests {
		handler := NewScaleHandler(
			test.input,
			schema.GroupVersion{Group: "apps", Version: "v1"},
			defaultMappings(),
		)
		subresourceEntries, err := handler.ToSubresource()
		if err != nil {
			t.Fatalf("test %q - expected no error but got %v", test.desc, err)
		}

		if !reflect.DeepEqual(subresourceEntries, test.expected) {
			t.Fatalf("test %q - expected output to be:\n%v\n\nbut got:\n%v", test.desc, test.expected, subresourceEntries)
		}
	}
}

func TestTransformingManagedFieldsToParent(t *testing.T) {
	tests := []struct {
		desc        string
		parent      []metav1.ManagedFieldsEntry
		subresource []metav1.ManagedFieldsEntry
		expected    []metav1.ManagedFieldsEntry
	}{
		{
			desc: "different-managers: apply -> update",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{},"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
		},
		{
			desc: "different-managers: apply -> apply",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{},"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
			},
		},
		{
			desc: "different-managers: update -> update",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{},"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
			},
		},
		{
			desc: "different-managers: update -> apply",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{},"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
			},
		},
		{
			desc: "same manager: apply -> apply",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{},"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "test",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
				{
					Manager:     "test",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
		},
		{
			desc: "same manager: update -> update",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{},"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "test",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
				{
					Manager:     "test",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
		},
		{
			desc: "same manager: update -> apply",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{},"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "test",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:     "test",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
			},
		},
		{
			desc: "same manager: apply -> update",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{},"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "test",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
				{
					Manager:     "test",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
		},
		{
			desc: "subresource doesn't own the path anymore",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:status":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
			},
		},
		{
			desc: "Subresource steals all the fields of the parent resource",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
		},
		{
			desc: "apply without stealing",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{},"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "autoscaling/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
				},
				{
					Manager:     "test",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{},"f:selector":{}}}`)},
				},
				{
					Manager:     "test",
					Operation:   metav1.ManagedFieldsOperationApply,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
		},
		{
			desc: "drops the entry if the api version is unknown",
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
				},
				{
					Manager:    "another-manager",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v10",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:    "scale",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "autoscaling/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:    "scale",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			handler := NewScaleHandler(
				test.parent,
				schema.GroupVersion{Group: "apps", Version: "v1"},
				defaultMappings(),
			)
			parentEntries, err := handler.ToParent(test.subresource)
			if err != nil {
				t.Fatalf("test: %q - expected no error but got %v", test.desc, err)
			}
			if !reflect.DeepEqual(parentEntries, test.expected) {
				t.Fatalf("test: %q - expected output to be:\n%v\n\nbut got:\n%v", test.desc, test.expected, parentEntries)
			}
		})
	}
}

func TestTransformingManagedFieldsToParentMultiVersion(t *testing.T) {
	tests := []struct {
		desc         string
		groupVersion schema.GroupVersion
		mappings     ResourcePathMappings
		parent       []metav1.ManagedFieldsEntry
		subresource  []metav1.ManagedFieldsEntry
		expected     []metav1.ManagedFieldsEntry
	}{
		{
			desc:         "multi-version",
			groupVersion: schema.GroupVersion{Group: "apps", Version: "v1"},
			mappings: ResourcePathMappings{
				"apps/v1": fieldpath.MakePathOrDie("spec", "the-replicas"),
				"apps/v2": fieldpath.MakePathOrDie("spec", "not-the-replicas"),
			},
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:the-replicas":{},"f:selector":{}}}`)},
				},
				{
					Manager:    "test-other",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v2",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:not-the-replicas":{},"f:selector":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
				{
					Manager:    "test-other",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "apps/v2",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "apps/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:the-replicas":{}}}`)},
					Subresource: "scale",
				},
			},
		},
		{
			desc:         "Custom resource without scale subresource, scaling a version with `scale`",
			groupVersion: schema.GroupVersion{Group: "mygroup", Version: "v1"},
			mappings: ResourcePathMappings{
				"mygroup/v1": fieldpath.MakePathOrDie("spec", "the-replicas"),
				"mygroup/v2": nil,
			},
			parent: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "mygroup/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:the-replicas":{},"f:selector":{}}}`)},
				},
				{
					Manager:    "test-other",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "mygroup/v2",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:test-other":{}}}`)},
				},
			},
			subresource: []metav1.ManagedFieldsEntry{
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "autoscaling/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:replicas":{}}}`)},
					Subresource: "scale",
				},
			},
			expected: []metav1.ManagedFieldsEntry{
				{
					Manager:    "test",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "mygroup/v1",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:selector":{}}}`)},
				},
				{
					Manager:    "test-other",
					Operation:  metav1.ManagedFieldsOperationApply,
					APIVersion: "mygroup/v2",
					FieldsType: "FieldsV1",
					FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:test-other":{}}}`)},
				},
				{
					Manager:     "scale",
					Operation:   metav1.ManagedFieldsOperationUpdate,
					APIVersion:  "mygroup/v1",
					FieldsType:  "FieldsV1",
					FieldsV1:    &metav1.FieldsV1{Raw: []byte(`{"f:spec":{"f:the-replicas":{}}}`)},
					Subresource: "scale",
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			handler := NewScaleHandler(
				test.parent,
				test.groupVersion,
				test.mappings,
			)
			parentEntries, err := handler.ToParent(test.subresource)
			if err != nil {
				t.Fatalf("test: %q - expected no error but got %v", test.desc, err)
			}
			if !reflect.DeepEqual(parentEntries, test.expected) {
				t.Fatalf("test: %q - expected output to be:\n%v\n\nbut got:\n%v", test.desc, test.expected, parentEntries)
			}
		})
	}
}

func defaultMappings() ResourcePathMappings {
	return ResourcePathMappings{
		"apps/v1": fieldpath.MakePathOrDie("spec", "replicas"),
	}
}
