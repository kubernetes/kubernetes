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

package rest

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TestWipeObjectMetaSystemFields validates that system populated fields are set on an object
func TestWipeObjectMetaSystemFields(t *testing.T) {
	resource := metav1.ObjectMeta{}
	WipeObjectMetaSystemFields(&resource)
	if !resource.CreationTimestamp.Time.IsZero() {
		t.Errorf("resource.CreationTimestamp is set")
	}
	if len(resource.UID) != 0 {
		t.Errorf("resource.UID is set")
	}
	if resource.DeletionTimestamp != nil {
		t.Errorf("resource.DeletionTimestamp is set")
	}
	if resource.DeletionGracePeriodSeconds != nil {
		t.Errorf("resource.DeletionGracePeriodSeconds is set")
	}
	if len(resource.SelfLink) != 0 {
		t.Errorf("resource.SelfLink is set")
	}
}

// TestFillObjectMetaSystemFields validates that system populated fields are set on an object
func TestFillObjectMetaSystemFields(t *testing.T) {
	resource := metav1.ObjectMeta{}
	FillObjectMetaSystemFields(&resource)
	if resource.CreationTimestamp.Time.IsZero() {
		t.Errorf("resource.CreationTimestamp is zero")
	}
	if len(resource.UID) == 0 {
		t.Errorf("resource.UID missing")
	}
}

// TestHasObjectMetaSystemFieldValues validates that true is returned if and only if all fields are populated
func TestHasObjectMetaSystemFieldValues(t *testing.T) {
	resource := metav1.ObjectMeta{}
	objMeta, err := meta.Accessor(&resource)
	if err != nil {
		t.Fatal(err)
	}
	if metav1.HasObjectMetaSystemFieldValues(objMeta) {
		t.Errorf("the resource does not have all fields yet populated, but incorrectly reports it does")
	}
	FillObjectMetaSystemFields(&resource)
	if !metav1.HasObjectMetaSystemFieldValues(objMeta) {
		t.Errorf("the resource does have all fields populated, but incorrectly reports it does not")
	}
}

func TestEnsureObjectNamespaceMatchesRequestNamespace(t *testing.T) {
	testcases := []struct {
		name        string
		reqNS       string
		objNS       string
		expectErr   bool
		expectObjNS string
	}{
		{
			name:        "cluster-scoped req, cluster-scoped obj",
			reqNS:       "",
			objNS:       "",
			expectErr:   false,
			expectObjNS: "",
		},
		{
			name:        "cluster-scoped req, namespaced obj",
			reqNS:       "",
			objNS:       "foo",
			expectErr:   false,
			expectObjNS: "", // no error, object is forced to cluster-scoped for backwards compatibility
		},
		{
			name:        "namespaced req, no-namespace obj",
			reqNS:       "foo",
			objNS:       "",
			expectErr:   false,
			expectObjNS: "foo", // no error, object is updated to match request for backwards compatibility
		},
		{
			name:        "namespaced req, matching obj",
			reqNS:       "foo",
			objNS:       "foo",
			expectErr:   false,
			expectObjNS: "foo",
		},
		{
			name:      "namespaced req, mis-matched obj",
			reqNS:     "foo",
			objNS:     "bar",
			expectErr: true,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			obj := metav1.ObjectMeta{Namespace: tc.objNS}
			err := EnsureObjectNamespaceMatchesRequestNamespace(tc.reqNS, &obj)
			if tc.expectErr {
				if err == nil {
					t.Fatal("expected err, got none")
				}
				return
			} else if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}

			if obj.Namespace != tc.expectObjNS {
				t.Fatalf("expected obj ns %q, got %q", tc.expectObjNS, obj.Namespace)
			}
		})
	}
}
