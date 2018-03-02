/*
Copyright 2015 The Kubernetes Authors.

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

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, extensions.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "daemonsets",
	}
	daemonSetStorage, statusStorage := NewREST(restOptions)
	return daemonSetStorage, statusStorage, server
}

func newValidDaemonSet() *extensions.DaemonSet {
	return &extensions.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: extensions.DaemonSetSpec{
			UpdateStrategy: extensions.DaemonSetUpdateStrategy{
				Type: extensions.OnDeleteDaemonSetStrategyType,
			},
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:                     "test",
							Image:                    "test_image",
							ImagePullPolicy:          api.PullIfNotPresent,
							TerminationMessagePolicy: api.TerminationMessageReadFile,
						},
					},
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
	}
}

var validDaemonSet = newValidDaemonSet()

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	ds := newValidDaemonSet()
	ds.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		ds,
		// invalid (invalid selector)
		&extensions.DaemonSet{
			Spec: extensions.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: map[string]string{}},
				Template: validDaemonSet.Spec.Template,
			},
		},
		// invalid update strategy
		&extensions.DaemonSet{
			Spec: extensions.DaemonSetSpec{
				Selector: validDaemonSet.Spec.Selector,
				Template: validDaemonSet.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestUpdate(
		// valid
		newValidDaemonSet(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.DaemonSet)
			object.Spec.Template.Spec.NodeSelector = map[string]string{"c": "d"}
			object.Spec.Template.Spec.DNSPolicy = api.DNSDefault
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.DaemonSet)
			object.Name = ""
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.DaemonSet)
			object.Spec.Template.Spec.RestartPolicy = api.RestartPolicyOnFailure
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestDelete(newValidDaemonSet())
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestGet(newValidDaemonSet())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestList(newValidDaemonSet())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		validDaemonSet,
		// matching labels
		[]labels.Set{
			{"a": "b"},
		},
		// not matching labels
		[]labels.Set{
			{"a": "c"},
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// notmatching fields
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
	expected := []string{"ds"}
	registrytest.AssertShortNames(t, storage, expected)
}

func TestUpdateStatus(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	ctx := genericapirequest.NewDefaultContext()

	key, _ := storage.KeyFunc(ctx, "foo")
	dsStart := newValidDaemonSet()
	if err := storage.Storage.Create(ctx, key, dsStart, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	dsIn := &extensions.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Status: extensions.DaemonSetStatus{
			Conditions: []extensions.DaemonSetCondition{
				{Status: "True"},
			},
		},
	}

	if _, _, err := statusStorage.Update(ctx, dsIn.Name, rest.DefaultUpdatedObjectInfo(dsIn), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	dsOut := obj.(*extensions.DaemonSet)
	// only compare relevant changes b/c of difference in metadata
	if !apiequality.Semantic.DeepEqual(dsIn.Status, dsOut.Status) {
		t.Errorf("unexpected object: %s", diff.ObjectDiff(dsIn.Status, dsOut.Status))
	}
}
