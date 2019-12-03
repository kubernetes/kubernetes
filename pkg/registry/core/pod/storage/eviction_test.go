/*
Copyright 2019 The Kubernetes Authors.

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

	policyv1beta1 "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/api/apitesting"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/client-go/kubernetes/fake"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func TestEviction(t *testing.T) {
	testcases := []struct {
		name     string
		pdbs     []runtime.Object
		eviction *policy.Eviction

		badNameInURL bool

		expectError   bool
		expectDeleted bool
	}{
		{
			name: "matching pdbs with no disruptions allowed",
			pdbs: []runtime.Object{&policyv1beta1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1beta1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1beta1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:    &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError: true,
		},
		{
			name: "matching pdbs with disruptions allowed",
			pdbs: []runtime.Object{&policyv1beta1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1beta1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1beta1.PodDisruptionBudgetStatus{DisruptionsAllowed: 1},
			}},
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectDeleted: true,
		},
		{
			name: "non-matching pdbs",
			pdbs: []runtime.Object{&policyv1beta1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1beta1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"b": "true"}}},
				Status:     policyv1beta1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectDeleted: true,
		},
		{
			name: "matching pdbs with disruptions allowed but bad name in Url",
			pdbs: []runtime.Object{&policyv1beta1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1beta1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1beta1.PodDisruptionBudgetStatus{DisruptionsAllowed: 1},
			}},
			badNameInURL: true,
			eviction:     &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:  true,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
			storage, _, _, server := newStorage(t)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			pod := validNewPod()
			pod.Labels = map[string]string{"a": "true"}
			pod.Spec.NodeName = "foo"

			if _, err := storage.Create(testContext, pod, nil, &metav1.CreateOptions{}); err != nil {
				t.Error(err)
			}

			client := fake.NewSimpleClientset(tc.pdbs...)
			evictionRest := newEvictionStorage(storage.Store, client.PolicyV1beta1())

			name := pod.Name
			if tc.badNameInURL {
				name += "bad-name"
			}
			_, err := evictionRest.Create(testContext, name, tc.eviction, nil, &metav1.CreateOptions{})
			if (err != nil) != tc.expectError {
				t.Errorf("expected error=%v, got %v", tc.expectError, err)
				return
			}
			if tc.badNameInURL {
				if err == nil {
					t.Error("expected error here, but got nil")
					return
				}
				if err.Error() != "name in URL does not match name in Eviction object" {
					t.Errorf("got unexpected error: %v", err)
				}
			}
			if tc.expectError {
				return
			}

			existingPod, err := storage.Get(testContext, pod.Name, &metav1.GetOptions{})
			if tc.expectDeleted {
				if !apierrors.IsNotFound(err) {
					t.Errorf("expected to be deleted, lookup returned %#v", existingPod)
				}
				return
			} else if apierrors.IsNotFound(err) {
				t.Errorf("expected graceful deletion, got %v", err)
				return
			}

			if err != nil {
				t.Errorf("%#v", err)
				return
			}

			if existingPod.(*api.Pod).DeletionTimestamp == nil {
				t.Errorf("expected gracefully deleted pod with deletionTimestamp set, got %#v", existingPod)
			}
		})
	}
}

type FailDeleteUpdateStorage struct {
	storage.Interface
}

func (f FailDeleteUpdateStorage) Delete(ctx context.Context, key string, out runtime.Object, precondition *storage.Preconditions, validateDeletion storage.ValidateObjectFunc) error {
	return storage.NewKeyNotFoundError(key, 0)
}

func (f FailDeleteUpdateStorage) GuaranteedUpdate(ctx context.Context, key string, ptrToType runtime.Object, ignoreNotFound bool,
	preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, suggestion ...runtime.Object) error {
	return storage.NewKeyNotFoundError(key, 0)
}

var scheme = runtime.NewScheme()
var codecs = serializer.NewCodecFactory(scheme)

func newFailDeleteUpdateStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 3,
		ResourcePrefix:          "pods",
	}
	storage, err := NewStorage(restOptions, nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	storage.Pod.Store.Storage = genericregistry.DryRunnableStorage{
		Storage: FailDeleteUpdateStorage{storage.Pod.Store.Storage.Storage},
		Codec:   apitesting.TestStorageCodec(codecs, examplev1.SchemeGroupVersion),
	}
	return storage.Pod, server
}

func TestEvictionDryRun(t *testing.T) {
	testcases := []struct {
		name            string
		evictionOptions *metav1.DeleteOptions
		requestOptions  *metav1.CreateOptions
		pdbs            []runtime.Object
	}{
		{
			name:            "just request-options",
			requestOptions:  &metav1.CreateOptions{DryRun: []string{"All"}},
			evictionOptions: &metav1.DeleteOptions{},
		},
		{
			name:            "just eviction-options",
			requestOptions:  &metav1.CreateOptions{},
			evictionOptions: &metav1.DeleteOptions{DryRun: []string{"All"}},
		},
		{
			name:            "both options",
			evictionOptions: &metav1.DeleteOptions{DryRun: []string{"All"}},
			requestOptions:  &metav1.CreateOptions{DryRun: []string{"All"}},
		},
		{
			name:            "with pdbs",
			evictionOptions: &metav1.DeleteOptions{DryRun: []string{"All"}},
			requestOptions:  &metav1.CreateOptions{DryRun: []string{"All"}},
			pdbs: []runtime.Object{&policyv1beta1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1beta1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1beta1.PodDisruptionBudgetStatus{DisruptionsAllowed: 1},
			}},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
			storage, server := newFailDeleteUpdateStorage(t)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			pod := validNewPod()
			pod.Labels = map[string]string{"a": "true"}
			pod.Spec.NodeName = "foo"
			if _, err := storage.Create(testContext, pod, nil, &metav1.CreateOptions{}); err != nil {
				t.Error(err)
			}

			client := fake.NewSimpleClientset(tc.pdbs...)
			evictionRest := newEvictionStorage(storage.Store, client.PolicyV1beta1())
			eviction := &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}, DeleteOptions: tc.evictionOptions}
			_, err := evictionRest.Create(testContext, pod.Name, eviction, nil, tc.requestOptions)
			if err != nil {
				t.Fatalf("Failed to run eviction: %v", err)
			}
		})
	}
}
