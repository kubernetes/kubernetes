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

package storage

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*ScaleREST, *etcdtesting.EtcdTestServer, storage.Interface, factory.DestroyFunc) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "controllers"}
	s, d := generic.NewRawStorage(etcdStorage)
	destroyFunc := func() {
		d()
		server.Terminate(t)
	}
	return NewStorage(restOptions).Scale, server, s, destroyFunc
}

var validPodTemplate = api.PodTemplate{
	Template: api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{"a": "b"},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "test",
					Image:           "test_image",
					ImagePullPolicy: api.PullIfNotPresent,
				},
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
	},
}

var validReplicas = int32(8)

var validControllerSpec = api.ReplicationControllerSpec{
	Replicas: validReplicas,
	Selector: validPodTemplate.Template.Labels,
	Template: &validPodTemplate.Template,
}

var validController = api.ReplicationController{
	ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
	Spec:       validControllerSpec,
}

var validScale = extensions.Scale{
	ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
	Spec: extensions.ScaleSpec{
		Replicas: validReplicas,
	},
	Status: extensions.ScaleStatus{
		Replicas: 0,
		Selector: &metav1.LabelSelector{
			MatchLabels: validPodTemplate.Template.Labels,
		},
	},
}

func TestGet(t *testing.T) {
	storage, _, si, destroyFunc := newStorage(t)
	defer destroyFunc()

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	key := "/controllers/test/foo"
	if err := si.Create(ctx, key, &validController, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	scale := obj.(*extensions.Scale)
	if scale.Spec.Replicas != validReplicas {
		t.Errorf("wrong replicas count expected: %d got: %d", validReplicas, scale.Spec.Replicas)
	}
}

func TestUpdate(t *testing.T) {
	storage, _, si, destroyFunc := newStorage(t)
	defer destroyFunc()

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	key := "/controllers/test/foo"
	if err := si.Create(ctx, key, &validController, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	replicas := int32(12)
	update := extensions.Scale{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec: extensions.ScaleSpec{
			Replicas: replicas,
		},
	}

	if _, _, err := storage.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update, api.Scheme)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	updated := obj.(*extensions.Scale)
	if updated.Spec.Replicas != replicas {
		t.Errorf("wrong replicas count expected: %d got: %d", replicas, updated.Spec.Replicas)
	}
}
