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
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/registry/rest/resttest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/registry/registrytest"

	// Ensure that admissionregistration package is initialized.
	_ "k8s.io/kubernetes/pkg/apis/admissionregistration/install"
)

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	configuration := validPolicyBinding()
	test.TestCreate(
		// valid
		configuration,
		// invalid
		newPolicyBinding(""),
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()

	test.TestUpdate(
		// valid
		validPolicyBinding(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*admissionregistration.ValidatingAdmissionPolicyBinding)
			object.Labels = map[string]string{"c": "d"}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*admissionregistration.ValidatingAdmissionPolicyBinding)
			object.Name = ""
			return object
		},
	)
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validPolicyBinding())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validPolicyBinding())
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestDelete(validPolicyBinding())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validPolicyBinding(),
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

func validPolicyBinding() *admissionregistration.ValidatingAdmissionPolicyBinding {
	return &admissionregistration.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: "replicalimit-policy.example.com",
			ParamRef: &admissionregistration.ParamRef{
				Name: "param-test",
			},
			MatchResources: &admissionregistration.MatchResources{
				MatchPolicy: func() *admissionregistration.MatchPolicyType {
					r := admissionregistration.MatchPolicyType("Exact")
					return &r
				}(),
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
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
		},
	}
}

func newPolicyBinding(name string) *admissionregistration.ValidatingAdmissionPolicyBinding {
	return &admissionregistration.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"foo": "bar"},
		},
		Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: "replicalimit-policy.example.com",
			ParamRef: &admissionregistration.ParamRef{
				Name: "param-test",
			},
			MatchResources: &admissionregistration.MatchResources{},
		},
	}
}

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, admissionregistration.Resource("validatingadmissionpolicybindings"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "validatingadmissionpolicybindings"}
	storage, err := NewREST(restOptions)
	storage.authorize = func(ctx context.Context) error {
		return nil
	}
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return storage, server
}

func TestCategories(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"api-extensions"}
	registrytest.AssertCategories(t, storage, expected)
}

func newSecureStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, admissionregistration.Resource("validatingadmissionpolicybindings"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "validatingadmissionpolicybindings"}
	storage, err := NewREST(restOptions)

	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return storage, server
}

func TestAuthorization(t *testing.T) {
	for _, tc := range []struct {
		name      string
		userInfo  user.Info
		expectErr bool
	}{
		{
			name:      "superuser",
			userInfo:  &user.DefaultInfo{Groups: []string{user.SystemPrivilegedGroup}},
			expectErr: false,
		},
		{
			name:      "authenticated",
			userInfo:  &user.DefaultInfo{Groups: []string{user.AllAuthenticated}},
			expectErr: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			storage, server := newSecureStorage(t)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()
			test := resttest.New(t, storage).ClusterScope()
			t.Run("create", func(t *testing.T) {
				ctx := request.WithUser(test.TestContext(), tc.userInfo)
				_, err := storage.Create(ctx, validPolicyBinding(), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
				if (err != nil) != tc.expectErr {
					t.Errorf("expected error: %v but got error: %v", tc.expectErr, err)
				}
			})
			t.Run("update", func(t *testing.T) {
				ctx := request.WithUser(test.TestContext(), tc.userInfo)
				obj := validPolicyBinding()
				_, _, err := storage.Update(ctx, obj.Name, rest.DefaultUpdatedObjectInfo(obj, func(ctx context.Context, newObj runtime.Object, oldObj runtime.Object) (transformedNewObj runtime.Object, err error) {
					object := oldObj.(*admissionregistration.ValidatingAdmissionPolicyBinding)
					object.Labels = map[string]string{"c": "d"}
					return object, nil
				}), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
				if (err != nil) != tc.expectErr {
					t.Errorf("expected error: %v but got error: %v", tc.expectErr, err)
				}
			})
		})
	}
}
