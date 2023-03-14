/*
Copyright 2022 The Kubernetes Authors.

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

package storage

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/resolver"
	"k8s.io/kubernetes/pkg/registry/registrytest"

	// Ensure that admissionregistration package is initialized.
	_ "k8s.io/kubernetes/pkg/apis/admissionregistration/install"
)

func TestCreate(t *testing.T) {
	storage, server := newInsecureStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	configuration := validValidatingAdmissionPolicy()
	test.TestCreate(
		// valid
		configuration,
		// invalid
		newValidatingAdmissionPolicy(""),
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newInsecureStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()

	test.TestUpdate(
		// valid
		validValidatingAdmissionPolicy(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*admissionregistration.ValidatingAdmissionPolicy)
			object.Labels = map[string]string{"c": "d"}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*admissionregistration.ValidatingAdmissionPolicy)
			object.Name = ""
			return object
		},
	)
}

func TestGet(t *testing.T) {
	storage, server := newInsecureStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validValidatingAdmissionPolicy())
}

func TestList(t *testing.T) {
	storage, server := newInsecureStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validValidatingAdmissionPolicy())
}

func TestDelete(t *testing.T) {
	storage, server := newInsecureStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestDelete(validValidatingAdmissionPolicy())
}

func TestWatch(t *testing.T) {
	storage, server := newInsecureStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validValidatingAdmissionPolicy(),
		[]labels.Set{},
		[]labels.Set{
			{"hoo": "bar"},
		},
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		[]fields.Set{
			{"metadata.name": "nomatch"},
		},
	)
}

func validValidatingAdmissionPolicy() *admissionregistration.ValidatingAdmissionPolicy {
	return &admissionregistration.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: admissionregistration.ValidatingAdmissionPolicySpec{
			FailurePolicy: func() *admissionregistration.FailurePolicyType {
				r := admissionregistration.FailurePolicyType("Fail")
				return &r
			}(),
			ParamKind: &admissionregistration.ParamKind{
				APIVersion: "rules.example.com/v1",
				Kind:       "ReplicaLimit",
			},
			Validations: []admissionregistration.Validation{
				{
					Expression: "object.spec.replicas <= params.maxReplicas",
				},
			},
			MatchConstraints: &admissionregistration.MatchResources{
				MatchPolicy: func() *admissionregistration.MatchPolicyType {
					r := admissionregistration.MatchPolicyType("Exact")
					return &r
				}(),
				ResourceRules: []admissionregistration.NamedRuleWithOperations{
					{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					},
				},
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
			},
		},
	}
}

func newValidatingAdmissionPolicy(name string) *admissionregistration.ValidatingAdmissionPolicy {
	ignore := admissionregistration.Ignore
	return &admissionregistration.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"foo": "bar"},
		},
		Spec: admissionregistration.ValidatingAdmissionPolicySpec{
			ParamKind: &admissionregistration.ParamKind{
				APIVersion: "rules.example.com/v1",
				Kind:       "ReplicaLimit",
			},
			Validations: []admissionregistration.Validation{
				{
					Expression: "object.spec.replicas <= params.maxReplicas",
				},
			},
			MatchConstraints: &admissionregistration.MatchResources{
				ResourceRules: []admissionregistration.NamedRuleWithOperations{
					{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					},
				},
			},
			FailurePolicy: &ignore,
		},
	}
}

func newInsecureStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	return newStorage(t, nil, nil)
}

func newStorage(t *testing.T, authorizer authorizer.Authorizer, resourceResolver resolver.ResourceResolver) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, admissionregistration.Resource("validatingadmissionpolicies"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "validatingadmissionpolicies"}
	storage, _, err := NewREST(restOptions, authorizer, resourceResolver)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return storage, server
}

func TestCategories(t *testing.T) {
	storage, server := newInsecureStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"api-extensions"}
	registrytest.AssertCategories(t, storage, expected)
}
