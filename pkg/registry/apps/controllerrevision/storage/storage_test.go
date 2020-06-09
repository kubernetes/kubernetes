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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	var (
		valid       = stripObjectMeta(newControllerRevision("validname", metav1.NamespaceDefault, newObject(), 0))
		badRevision = stripObjectMeta(newControllerRevision("validname", "validns", newObject(), -1))
		emptyName   = stripObjectMeta(newControllerRevision("", "validns", newObject(), 0))
		invalidName = stripObjectMeta(newControllerRevision("NoUppercaseOrSpecialCharsLike=Equals", "validns", newObject(), 0))
		emptyNs     = stripObjectMeta(newControllerRevision("validname", "", newObject(), 100))
		invalidNs   = stripObjectMeta(newControllerRevision("validname", "NoUppercaseOrSpecialCharsLike=Equals", newObject(), 100))
		nilData     = stripObjectMeta(newControllerRevision("validname", "validns", nil, 0))
	)
	test.TestCreate(
		valid,
		badRevision,
		emptyName,
		invalidName,
		emptyNs,
		invalidNs,
		nilData)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)

	addLabel := func(obj runtime.Object) runtime.Object {
		rev := obj.(*apps.ControllerRevision)
		update := &apps.ControllerRevision{
			ObjectMeta: rev.ObjectMeta,
			Data:       rev.Data,
			Revision:   rev.Revision,
		}
		update.ObjectMeta.Labels = map[string]string{"foo": "bar"}
		return update
	}

	updateData := func(obj runtime.Object) runtime.Object {
		rev := obj.(*apps.ControllerRevision)
		modified := newObject()
		ss := modified.(*apps.StatefulSet)
		ss.Name = "cde"
		update := &apps.ControllerRevision{
			ObjectMeta: rev.ObjectMeta,
			Data:       ss,
			Revision:   rev.Revision + 1,
		}
		return update
	}

	test.TestUpdate(stripObjectMeta(newControllerRevision("validname", metav1.NamespaceDefault, newObject(), 0)),
		addLabel,
		updateData)
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestGet(newControllerRevision("valid", metav1.NamespaceDefault, newObject(), 0))
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestList(newControllerRevision("valid", metav1.NamespaceDefault, newObject(), 0))
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestDelete(newControllerRevision("valid", metav1.NamespaceDefault, newObject(), 0))
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		newControllerRevision("valid", metav1.NamespaceDefault, newObject(), 0),
		[]labels.Set{
			{"foo": "bar"},
		},
		[]labels.Set{
			{"hoo": "baz"},
		},
		[]fields.Set{
			{"metadata.name": "valid"},
		},
		[]fields.Set{
			{"metadata.name": "nomatch"},
		},
	)
}

func newControllerRevision(name, namespace string, data runtime.Object, revision int64) *apps.ControllerRevision {
	return &apps.ControllerRevision{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    map[string]string{"foo": "bar"},
		},
		Data:     data,
		Revision: revision,
	}
}

func stripObjectMeta(revision *apps.ControllerRevision) *apps.ControllerRevision {
	revision.ObjectMeta = metav1.ObjectMeta{}
	return revision
}

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, apps.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "controllerrevisions"}
	storage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return storage, server
}

func newObject() runtime.Object {
	return &apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: apps.StatefulSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			Template: api.PodTemplateSpec{
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
			},
		},
	}
}
