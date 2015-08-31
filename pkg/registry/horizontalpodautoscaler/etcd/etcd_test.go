/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package etcd

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/expapi"
	// Ensure that expapi/v1 package is initialized.
	_ "k8s.io/kubernetes/pkg/expapi/v1"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"

	"github.com/coreos/go-etcd/etcd"
)

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t)
	return NewREST(etcdStorage), fakeClient
}

func validNewHorizontalPodAutoscaler(name string) *expapi.HorizontalPodAutoscaler {
	return &expapi.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: expapi.HorizontalPodAutoscalerSpec{
			ScaleRef: &expapi.SubresourceReference{
				Subresource: "scale",
			},
			MinCount: 1,
			MaxCount: 5,
			Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.8")},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	autoscaler := validNewHorizontalPodAutoscaler("foo")
	autoscaler.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		autoscaler,
		func(ctx api.Context, obj runtime.Object) error {
			return registrytest.SetObject(fakeClient, storage.KeyFunc, ctx, obj)
		},
		func(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
			return registrytest.GetObject(fakeClient, storage.KeyFunc, storage.NewFunc, ctx, obj)
		},
		// invalid
		&expapi.HorizontalPodAutoscaler{},
	)
}

func TestUpdate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	test.TestUpdate(
		// valid
		validNewHorizontalPodAutoscaler("foo"),
		func(ctx api.Context, obj runtime.Object) error {
			return registrytest.SetObject(fakeClient, storage.KeyFunc, ctx, obj)
		},
		func(resourceVersion uint64) {
			registrytest.SetResourceVersion(fakeClient, resourceVersion)
		},
		func(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
			return registrytest.GetObject(fakeClient, storage.KeyFunc, storage.NewFunc, ctx, obj)
		},
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*expapi.HorizontalPodAutoscaler)
			object.Spec.MaxCount = object.Spec.MaxCount + 1
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	autoscaler := validNewHorizontalPodAutoscaler("foo2")
	key, _ := storage.KeyFunc(ctx, "foo2")
	key = etcdtest.AddPrefix(key)
	createFn := func() runtime.Object {
		fakeClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Value:         runtime.EncodeOrDie(testapi.Codec(), autoscaler),
					ModifiedIndex: 1,
				},
			},
		}
		return autoscaler
	}
	gracefulSetFn := func() bool {
		if fakeClient.Data[key].R.Node == nil {
			return false
		}
		return fakeClient.Data[key].R.Node.TTL == 30
	}
	test.TestDelete(createFn, gracefulSetFn)
}

func TestGet(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	autoscaler := validNewHorizontalPodAutoscaler("foo")
	test.TestGet(autoscaler)
}

func TestList(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	key := etcdtest.AddPrefix(storage.KeyRootFunc(test.TestContext()))
	autoscaler := validNewHorizontalPodAutoscaler("foo")
	test.TestList(
		autoscaler,
		func(objects []runtime.Object) []runtime.Object {
			return registrytest.SetObjectsForKey(fakeClient, key, objects)
		},
		func(resourceVersion uint64) {
			registrytest.SetResourceVersion(fakeClient, resourceVersion)
		})
}
