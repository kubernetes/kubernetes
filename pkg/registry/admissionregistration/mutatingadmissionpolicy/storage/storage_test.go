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
	configuration := validMutatingAdmissionPolicy()
	test.TestCreate(
		// valid
		configuration,
		// invalid
		newMutatingAdmissionPolicy(""),
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newInsecureStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()

	test.TestUpdate(
		// valid
		validMutatingAdmissionPolicy(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*admissionregistration.MutatingAdmissionPolicy)
			object.Labels = map[string]string{"c": "d"}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*admissionregistration.MutatingAdmissionPolicy)
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
	test.TestGet(validMutatingAdmissionPolicy())
}

func TestList(t *testing.T) {
	storage, server := newInsecureStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validMutatingAdmissionPolicy())
}

func TestDelete(t *testing.T) {
	storage, server := newInsecureStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestDelete(validMutatingAdmissionPolicy())
}

func TestWatch(t *testing.T) {
	storage, server := newInsecureStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validMutatingAdmissionPolicy(),
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

func validMutatingAdmissionPolicy() *admissionregistration.MutatingAdmissionPolicy {
	applyConfigurationPatchType := admissionregistration.ApplyConfigurationPatchType
	return &admissionregistration.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: admissionregistration.MutatingAdmissionPolicySpec{
			FailurePolicy: func() *admissionregistration.FailurePolicyType {
				r := admissionregistration.FailurePolicyType("Fail")
				return &r
			}(),
			ParamKind: &admissionregistration.ParamKind{
				APIVersion: "rules.example.com/v1",
				Kind:       "ReplicaLimit",
			},
			Mutations: []admissionregistration.Mutation{
				{
					Expression: `Object{
							spec: Object.spec{
								replicas: object.spec.replicas % 2 == 0?object.spec.replicas + 1:object.spec.replicas
							}
						}`,
					PatchType: &applyConfigurationPatchType,
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

func newMutatingAdmissionPolicy(name string) *admissionregistration.MutatingAdmissionPolicy {
	ignore := admissionregistration.Ignore
	return &admissionregistration.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"foo": "bar"},
		},
		Spec: admissionregistration.MutatingAdmissionPolicySpec{
			ParamKind: &admissionregistration.ParamKind{
				APIVersion: "rules.example.com/v1",
				Kind:       "ReplicaLimit",
			},
			Mutations: []admissionregistration.Mutation{
				{
					// TODO cici: update the expression
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
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, admissionregistration.Resource("mutatingadmissionpolicies"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "mutatingadmissionpolicies"}
	storage, err := NewREST(restOptions, authorizer, resourceResolver)
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
