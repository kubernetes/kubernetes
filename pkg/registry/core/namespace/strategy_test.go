/*
Copyright 2014 The Kubernetes Authors.

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

package namespace

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"

	// install all api groups for testing
	_ "k8s.io/kubernetes/pkg/api/testapi"
)

func TestNamespaceStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Errorf("Namespaces should not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("Namespaces should not allow create on update")
	}
	namespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"},
		Status:     api.NamespaceStatus{Phase: api.NamespaceTerminating},
	}
	Strategy.PrepareForCreate(ctx, namespace)
	if namespace.Status.Phase != api.NamespaceActive {
		t.Errorf("Namespaces do not allow setting phase on create")
	}
	if len(namespace.Spec.Finalizers) != 1 || namespace.Spec.Finalizers[0] != api.FinalizerKubernetes {
		t.Errorf("Prepare For Create should have added kubernetes finalizer")
	}
	errs := Strategy.Validate(ctx, namespace)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	invalidNamespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", ResourceVersion: "4"},
	}
	// ensure we copy spec.finalizers from old to new
	Strategy.PrepareForUpdate(ctx, invalidNamespace, namespace)
	if len(invalidNamespace.Spec.Finalizers) != 1 || invalidNamespace.Spec.Finalizers[0] != api.FinalizerKubernetes {
		t.Errorf("PrepareForUpdate should have preserved old.spec.finalizers")
	}
	errs = Strategy.ValidateUpdate(ctx, invalidNamespace, namespace)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	if invalidNamespace.ResourceVersion != "4" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}

func TestNamespaceStatusStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if StatusStrategy.NamespaceScoped() {
		t.Errorf("Namespaces should not be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("Namespaces should not allow create on update")
	}
	now := metav1.Now()
	oldNamespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10", DeletionTimestamp: &now},
		Spec:       api.NamespaceSpec{Finalizers: []api.FinalizerName{"kubernetes"}},
		Status:     api.NamespaceStatus{Phase: api.NamespaceActive},
	}
	namespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "9", DeletionTimestamp: &now},
		Status:     api.NamespaceStatus{Phase: api.NamespaceTerminating},
	}
	StatusStrategy.PrepareForUpdate(ctx, namespace, oldNamespace)
	if namespace.Status.Phase != api.NamespaceTerminating {
		t.Errorf("Namespace status updates should allow change of phase: %v", namespace.Status.Phase)
	}
	if len(namespace.Spec.Finalizers) != 1 || namespace.Spec.Finalizers[0] != api.FinalizerKubernetes {
		t.Errorf("PrepareForUpdate should have preserved old finalizers")
	}
	errs := StatusStrategy.ValidateUpdate(ctx, namespace, oldNamespace)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
	if namespace.ResourceVersion != "9" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}

func TestNamespaceFinalizeStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if FinalizeStrategy.NamespaceScoped() {
		t.Errorf("Namespaces should not be namespace scoped")
	}
	if FinalizeStrategy.AllowCreateOnUpdate() {
		t.Errorf("Namespaces should not allow create on update")
	}
	oldNamespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"},
		Spec:       api.NamespaceSpec{Finalizers: []api.FinalizerName{"kubernetes", "example.com/org"}},
		Status:     api.NamespaceStatus{Phase: api.NamespaceActive},
	}
	namespace := &api.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "9"},
		Spec:       api.NamespaceSpec{Finalizers: []api.FinalizerName{"example.com/foo"}},
		Status:     api.NamespaceStatus{Phase: api.NamespaceTerminating},
	}
	FinalizeStrategy.PrepareForUpdate(ctx, namespace, oldNamespace)
	if namespace.Status.Phase != api.NamespaceActive {
		t.Errorf("finalize updates should not allow change of phase: %v", namespace.Status.Phase)
	}
	if len(namespace.Spec.Finalizers) != 1 || string(namespace.Spec.Finalizers[0]) != "example.com/foo" {
		t.Errorf("PrepareForUpdate should have modified finalizers")
	}
	errs := StatusStrategy.ValidateUpdate(ctx, namespace, oldNamespace)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
	if namespace.ResourceVersion != "9" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		"v1",
		"Namespace",
		NamespaceToSelectableFields(&api.Namespace{}),
		map[string]string{"name": "metadata.name"},
	)
}
