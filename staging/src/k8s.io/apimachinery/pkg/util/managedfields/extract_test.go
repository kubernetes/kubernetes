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
	"testing"

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/structured-merge-diff/v4/typed"

	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeschema "k8s.io/apimachinery/pkg/runtime/schema"
)

func TestExtractInto(t *testing.T) {
	one := int32(1)
	parser, err := typed.NewParser(schemaYAML)
	if err != nil {
		t.Fatalf("Failed to parse schema: %v", err)
	}
	cases := []struct {
		name          string
		obj           runtime.Object
		objType       typed.ParseableType
		managedFields []metav1.ManagedFieldsEntry // written to object before test is run
		fieldManager  string
		expectedOut   interface{}
		subresource   string
	}{
		{
			name:    "unstructured, no matching manager",
			obj:     &unstructured.Unstructured{Object: map[string]interface{}{"spec": map[string]interface{}{"replicas": 1}}},
			objType: parser.Type("io.k8s.api.apps.v1.Deployment"),
			managedFields: []metav1.ManagedFieldsEntry{
				applyFieldsEntry("mgr999", `{ "f:spec": { "f:replicas": {}}}`, ""),
			},
			fieldManager: "mgr1",
			expectedOut:  map[string]interface{}{},
		},
		{
			name:    "unstructured, one manager",
			obj:     &unstructured.Unstructured{Object: map[string]interface{}{"spec": map[string]interface{}{"replicas": 1}}},
			objType: parser.Type("io.k8s.api.apps.v1.Deployment"),
			managedFields: []metav1.ManagedFieldsEntry{
				applyFieldsEntry("mgr1", `{ "f:spec": { "f:replicas": {}}}`, ""),
			},
			fieldManager: "mgr1",
			expectedOut:  map[string]interface{}{"spec": map[string]interface{}{"replicas": 1}},
		},
		{
			name:    "unstructured, multiple manager",
			obj:     &unstructured.Unstructured{Object: map[string]interface{}{"spec": map[string]interface{}{"paused": true}}},
			objType: parser.Type("io.k8s.api.apps.v1.Deployment"),
			managedFields: []metav1.ManagedFieldsEntry{
				applyFieldsEntry("mgr1", `{ "f:spec": { "f:replicas": {}}}`, ""),
				applyFieldsEntry("mgr2", `{ "f:spec": { "f:paused": {}}}`, ""),
			},
			fieldManager: "mgr2",
			expectedOut:  map[string]interface{}{"spec": map[string]interface{}{"paused": true}},
		},
		{
			name:    "structured, no matching manager",
			obj:     &fakeDeployment{Spec: fakeDeploymentSpec{Replicas: &one}},
			objType: parser.Type("io.k8s.api.apps.v1.Deployment"),
			managedFields: []metav1.ManagedFieldsEntry{
				applyFieldsEntry("mgr999", `{ "f:spec": { "f:replicas": {}}}`, ""),
			},
			fieldManager: "mgr1",
			expectedOut:  map[string]interface{}{},
		},
		{
			name:    "structured, one manager",
			obj:     &fakeDeployment{Spec: fakeDeploymentSpec{Replicas: &one}},
			objType: parser.Type("io.k8s.api.apps.v1.Deployment"),
			managedFields: []metav1.ManagedFieldsEntry{
				applyFieldsEntry("mgr1", `{ "f:spec": { "f:replicas": {}}}`, ""),
			},
			fieldManager: "mgr1",
			expectedOut:  map[string]interface{}{"spec": map[string]interface{}{"replicas": int64(1)}},
		},
		{
			name:    "structured, multiple manager",
			obj:     &fakeDeployment{Spec: fakeDeploymentSpec{Replicas: &one, Paused: true}},
			objType: parser.Type("io.k8s.api.apps.v1.Deployment"),
			managedFields: []metav1.ManagedFieldsEntry{
				applyFieldsEntry("mgr1", `{ "f:spec": { "f:replicas": {}}}`, ""),
				applyFieldsEntry("mgr2", `{ "f:spec": { "f:paused": {}}}`, ""),
			},
			fieldManager: "mgr2",
			expectedOut:  map[string]interface{}{"spec": map[string]interface{}{"paused": true}},
		},
		{
			name:    "subresource",
			obj:     &fakeDeployment{Status: fakeDeploymentStatus{Replicas: &one}},
			objType: parser.Type("io.k8s.api.apps.v1.Deployment"),
			managedFields: []metav1.ManagedFieldsEntry{
				applyFieldsEntry("mgr1", `{ "f:status": { "f:replicas": {}}}`, "status"),
			},
			fieldManager: "mgr1",
			expectedOut:  map[string]interface{}{"status": map[string]interface{}{"replicas": int64(1)}},
			subresource:  "status",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := map[string]interface{}{}
			accessor, err := meta.Accessor(tc.obj)
			if err != nil {
				t.Fatalf("Error accessing object: %v", err)
			}
			accessor.SetManagedFields(tc.managedFields)
			err = ExtractInto(tc.obj, tc.objType, tc.fieldManager, &out, tc.subresource)
			if err != nil {
				t.Fatalf("Unexpected extract error: %v", err)
			}
			if !equality.Semantic.DeepEqual(out, tc.expectedOut) {
				t.Fatalf("Expected output did not match actual output: %s", cmp.Diff(out, tc.expectedOut))
			}
		})
	}
}

func applyFieldsEntry(fieldManager string, fieldsJSON string, subresource string) metav1.ManagedFieldsEntry {
	return metav1.ManagedFieldsEntry{
		Manager:     fieldManager,
		Operation:   metav1.ManagedFieldsOperationApply,
		APIVersion:  "v1",
		FieldsType:  "FieldsV1",
		FieldsV1:    &metav1.FieldsV1{Raw: []byte(fieldsJSON)},
		Subresource: subresource,
	}
}

type fakeDeployment struct {
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              fakeDeploymentSpec   `json:"spec"`
	Status            fakeDeploymentStatus `json:"status"`
}

type fakeDeploymentSpec struct {
	Replicas *int32 `json:"replicas"`
	Paused   bool   `json:"paused,omitempty"`
}

type fakeDeploymentStatus struct {
	Replicas *int32 `json:"replicas"`
}

func (o *fakeDeployment) GetObjectMeta() metav1.ObjectMeta {
	return o.ObjectMeta
}
func (o *fakeDeployment) GetObjectKind() runtimeschema.ObjectKind {
	return runtimeschema.EmptyObjectKind
}
func (o *fakeDeployment) DeepCopyObject() runtime.Object {
	return o
}

// trimmed up schema for test purposes
const schemaYAML = typed.YAMLObject(`types:
- name: io.k8s.api.apps.v1.Deployment
  map:
    fields:
    - name: apiVersion
      type:
        scalar: string
    - name: kind
      type:
        scalar: string
    - name: metadata
      type:
        namedType: io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta
    - name: spec
      type:
        namedType: io.k8s.api.apps.v1.DeploymentSpec
    - name: status
      type:
        namedType: io.k8s.api.apps.v1.DeploymentStatus
- name: io.k8s.api.apps.v1.DeploymentSpec
  map:
    fields:
    - name: paused
      type:
        scalar: boolean
    - name: replicas
      type:
        scalar: numeric
- name: io.k8s.api.apps.v1.DeploymentStatus
  map:
    fields:
    - name: replicas
      type:
        scalar: numeric
- name: io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta
  map:
    fields:
    - name: creationTimestamp
      type:
        namedType: io.k8s.apimachinery.pkg.apis.meta.v1.Time
    - name: managedFields
      type:
        list:
          elementType:
            namedType: io.k8s.apimachinery.pkg.apis.meta.v1.ManagedFieldsEntry
          elementRelationship: atomic
- name: io.k8s.apimachinery.pkg.apis.meta.v1.ManagedFieldsEntry
  map:
    fields:
    - name: apiVersion
      type:
        scalar: string
    - name: fieldsType
      type:
        scalar: string
    - name: fieldsV1
      type:
        namedType: io.k8s.apimachinery.pkg.apis.meta.v1.FieldsV1
    - name: manager
      type:
        scalar: string
    - name: operation
      type:
        scalar: string
    - name: time
      type:
        namedType: io.k8s.apimachinery.pkg.apis.meta.v1.Time
    - name: subresource
      type:
        scalar: string
- name: io.k8s.apimachinery.pkg.apis.meta.v1.FieldsV1
  map:
    elementType:
      scalar: untyped
      list:
        elementType:
          namedType: __untyped_atomic_
        elementRelationship: atomic
      map:
        elementType:
          namedType: __untyped_atomic_
        elementRelationship: atomic
- name: io.k8s.apimachinery.pkg.apis.meta.v1.Time
  scalar: untyped
- name: __untyped_atomic_
  scalar: untyped
  list:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
  map:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
`)
