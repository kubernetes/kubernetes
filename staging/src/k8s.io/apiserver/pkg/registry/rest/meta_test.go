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
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/apis/example"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// TestFillObjectMetaSystemFields validates that system populated fields are set on an object
func TestFillObjectMetaSystemFields(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	resource := metav1.ObjectMeta{}
	FillObjectMetaSystemFields(ctx, &resource)
	if resource.CreationTimestamp.Time.IsZero() {
		t.Errorf("resource.CreationTimestamp is zero")
	} else if len(resource.UID) == 0 {
		t.Errorf("resource.UID missing")
	}
	// verify we can inject a UID
	uid := uuid.NewUUID()
	ctx = genericapirequest.WithUID(ctx, uid)
	resource = metav1.ObjectMeta{}
	FillObjectMetaSystemFields(ctx, &resource)
	if resource.UID != uid {
		t.Errorf("resource.UID expected: %v, actual: %v", uid, resource.UID)
	}
}

// TestHasObjectMetaSystemFieldValues validates that true is returned if and only if all fields are populated
func TestHasObjectMetaSystemFieldValues(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	resource := metav1.ObjectMeta{}
	objMeta, err := meta.Accessor(&resource)
	if err != nil {
		t.Fatal(err)
	}
	if metav1.HasObjectMetaSystemFieldValues(objMeta) {
		t.Errorf("the resource does not have all fields yet populated, but incorrectly reports it does")
	}
	FillObjectMetaSystemFields(ctx, &resource)
	if !metav1.HasObjectMetaSystemFieldValues(objMeta) {
		t.Errorf("the resource does have all fields populated, but incorrectly reports it does not")
	}
}

// TestValidNamespace validates that namespace rules are enforced on a resource prior to create or update
func TestValidNamespace(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	namespace, _ := genericapirequest.NamespaceFrom(ctx)
	// TODO: use some genericapiserver type here instead of clientapiv1
	resource := example.Pod{}
	if !ValidNamespace(ctx, &resource.ObjectMeta) {
		t.Fatalf("expected success")
	}
	if namespace != resource.Namespace {
		t.Fatalf("expected resource to have the default namespace assigned during validation")
	}
	resource = example.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "other"}}
	if ValidNamespace(ctx, &resource.ObjectMeta) {
		t.Fatalf("Expected error that resource and context errors do not match because resource has different namespace")
	}
	ctx = genericapirequest.NewContext()
	if ValidNamespace(ctx, &resource.ObjectMeta) {
		t.Fatalf("Expected error that resource and context errors do not match since context has no namespace")
	}

	ctx = genericapirequest.NewContext()
	ns := genericapirequest.NamespaceValue(ctx)
	if ns != "" {
		t.Fatalf("Expected the empty string")
	}
}
