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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/util"
)

func newStorage(t *testing.T) (*DeploymentStorage, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "extensions")
	deploymentStorage := NewStorage(etcdStorage, generic.UndecoratedStorage)
	return &deploymentStorage, server
}

var namespace = "foo-namespace"
var name = "foo-deployment"

func validNewDeployment() *extensions.Deployment {
	return &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: extensions.DeploymentSpec{
			Selector: map[string]string{"a": "b"},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
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
			UniqueLabelKey: "my-label",
			Replicas:       7,
		},
		Status: extensions.DeploymentStatus{
			Replicas: 5,
		},
	}
}

var validDeployment = *validNewDeployment()

func validNewScale() *extensions.Scale {
	return &extensions.Scale{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespace},
		Spec: extensions.ScaleSpec{
			Replicas: validDeployment.Spec.Replicas,
		},
		Status: extensions.ScaleStatus{
			Replicas: validDeployment.Status.Replicas,
			Selector: validDeployment.Spec.Template.Labels,
		},
	}
}

var validScale = *validNewScale()

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Deployment.Etcd)
	deployment := validNewDeployment()
	deployment.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		deployment,
		// invalid (invalid selector)
		&extensions.Deployment{
			Spec: extensions.DeploymentSpec{
				Selector: map[string]string{},
				Template: validDeployment.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Deployment.Etcd)
	test.TestUpdate(
		// valid
		validNewDeployment(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Deployment)
			object.Spec.Template.Spec.NodeSelector = map[string]string{"c": "d"}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Deployment)
			object.UID = "newUID"
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Deployment)
			object.Name = ""
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Deployment)
			object.Spec.Template.Spec.RestartPolicy = api.RestartPolicyOnFailure
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Deployment)
			object.Spec.Selector = map[string]string{}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Deployment.Etcd)
	test.TestDelete(validNewDeployment())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Deployment.Etcd)
	test.TestGet(validNewDeployment())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Deployment.Etcd)
	test.TestList(validNewDeployment())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Deployment.Etcd)
	test.TestWatch(
		validNewDeployment(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"a": "c"},
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": name},
		},
		// not matchin fields
		[]fields.Set{
			{"metadata.name": "bar"},
			{"name": name},
		},
	)
}

func TestScaleGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)

	ctx := api.WithNamespace(api.NewContext(), namespace)
	key := etcdtest.AddPrefix("/deployments/" + namespace + "/" + name)
	if err := storage.Deployment.Storage.Set(ctx, key, &validDeployment, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expect := &validScale
	obj, err := storage.Scale.Get(ctx, name)
	scale := obj.(*extensions.Scale)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e, a := expect, scale; !api.Semantic.DeepDerivative(e, a) {
		t.Errorf("unexpected scale: %s", util.ObjectDiff(e, a))
	}
}

func TestScaleUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	ctx := api.WithNamespace(api.NewContext(), namespace)
	key := etcdtest.AddPrefix("/deployments/" + namespace + "/" + name)
	if err := storage.Deployment.Storage.Set(ctx, key, &validDeployment, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	replicas := 12
	update := extensions.Scale{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespace},
		Spec: extensions.ScaleSpec{
			Replicas: replicas,
		},
	}

	if _, _, err := storage.Scale.Update(ctx, &update); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Scale.Get(ctx, name)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	deployment := obj.(*extensions.Scale)
	if deployment.Spec.Replicas != replicas {
		t.Errorf("wrong replicas count expected: %d got: %d", replicas, deployment.Spec.Replicas)
	}
}

func TestStatusUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)

	ctx := api.WithNamespace(api.NewContext(), namespace)
	key := etcdtest.AddPrefix("/deployments/" + namespace + "/" + name)
	if err := storage.Deployment.Storage.Set(ctx, key, &validDeployment, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	update := extensions.Deployment{
		ObjectMeta: validDeployment.ObjectMeta,
		Spec: extensions.DeploymentSpec{
			Replicas: 100,
		},
		Status: extensions.DeploymentStatus{
			Replicas: 100,
		},
	}

	if _, _, err := storage.Status.Update(ctx, &update); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Deployment.Get(ctx, name)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	deployment := obj.(*extensions.Deployment)
	if deployment.Spec.Replicas != 7 {
		t.Errorf("we expected .spec.replicas to not be updated but it was updated to %v", deployment.Spec.Replicas)
	}
	if deployment.Status.Replicas != 100 {
		t.Errorf("we expected .status.replicas to be updated to 100 but it was %v", deployment.Status.Replicas)
	}
}
