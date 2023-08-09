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

package storage

import (
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/networking"
	_ "k8s.io/kubernetes/pkg/apis/networking/install"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, networking.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "networkpolicies",
	}
	rest, status, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return rest, status, server
}

func validNetworkPolicy() *networking.NetworkPolicy {
	return &networking.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
		},
		Spec: networking.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{
				MatchLabels: map[string]string{"label-1": "value-1"},
			},
			Ingress: []networking.NetworkPolicyIngressRule{
				{
					From: []networking.NetworkPolicyPeer{
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{"label-2": "value-2"},
							},
						},
					},
				},
			},
		},
		Status: networking.NetworkPolicyStatus{},
	}
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	np := validNetworkPolicy()
	np.ObjectMeta = metav1.ObjectMeta{GenerateName: "foo-"}
	test.TestCreate(
		// valid
		np,
		// invalid
		&networking.NetworkPolicy{
			ObjectMeta: metav1.ObjectMeta{Name: "name with spaces"},
		},
	)
}

func TestUpdate(t *testing.T) {
	protocolICMP := api.Protocol("ICMP")
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestUpdate(
		// valid
		validNetworkPolicy(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*networking.NetworkPolicy)
			object.Spec.Ingress = []networking.NetworkPolicyIngressRule{
				{
					From: []networking.NetworkPolicyPeer{
						{
							IPBlock: &networking.IPBlock{
								CIDR:   "192.168.0.0/16",
								Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
							},
						},
					},
				},
			}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*networking.NetworkPolicy)
			object.Spec.Ingress = []networking.NetworkPolicyIngressRule{
				{
					Ports: []networking.NetworkPolicyPort{
						{
							Protocol: &protocolICMP,
							Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 80},
						},
					},
				},
			}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestDelete(validNetworkPolicy())
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestGet(validNetworkPolicy())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestList(validNetworkPolicy())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		validNetworkPolicy(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
			{"name": "foo"},
		},
	)
}

func TestShortNames(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"netpol"}
	registrytest.AssertShortNames(t, storage, expected)
}

func TestStatusUpdate(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkPolicyStatus, true)()
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/networkpolicies/" + metav1.NamespaceDefault + "/foo"
	validNetPolObject := validNetworkPolicy()
	if err := storage.Storage.Create(ctx, key, validNetPolObject, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get netpol: %v", err)
	}

	obtainedNetPol := obj.(*networking.NetworkPolicy)

	transition := time.Now().Add(-5 * time.Minute)
	update := networking.NetworkPolicy{
		ObjectMeta: obtainedNetPol.ObjectMeta,
		Spec:       obtainedNetPol.Spec,
		Status: networking.NetworkPolicyStatus{
			Conditions: []metav1.Condition{
				{
					Type:   string(networking.NetworkPolicyConditionStatusAccepted),
					Status: metav1.ConditionTrue,
					LastTransitionTime: metav1.Time{
						Time: transition,
					},
					Reason:             "RuleApplied",
					Message:            "rule was successfully applied",
					ObservedGeneration: 2,
				},
			},
		},
	}

	if _, _, err := statusStorage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err = storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	netpol := obj.(*networking.NetworkPolicy)
	if len(netpol.Status.Conditions) != 1 {
		t.Fatalf("we expected 1 condition to exist in status but %d occurred", len(netpol.Status.Conditions))
	}

	condition := netpol.Status.Conditions[0]
	if condition.Type != string(networking.NetworkPolicyConditionStatusAccepted) {
		t.Errorf("we expected condition type to be %s but %s was returned", string(networking.NetworkPolicyConditionStatusAccepted), condition.Type)
	}

	if condition.Status != metav1.ConditionTrue {
		t.Errorf("we expected condition status to be true, but it returned false")
	}

	if condition.Reason != "RuleApplied" {
		t.Errorf("we expected condition reason to be RuleApplied, but %s was returned", condition.Reason)
	}

	if condition.Message != "rule was successfully applied" {
		t.Errorf("we expected message to be 'rule was successfully applied', but %s was returned", condition.Message)
	}

	if condition.ObservedGeneration != 2 {
		t.Errorf("we expected observedGeneration to be 2, but %d was returned", condition.ObservedGeneration)
	}

}
