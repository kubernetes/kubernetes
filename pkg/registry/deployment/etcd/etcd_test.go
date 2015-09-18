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
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"
)

func newStorage(t *testing.T) (*DeploymentStorage, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t, "experimental")
	deploymentStorage := NewStorage(etcdStorage)
	return &deploymentStorage, fakeClient
}

var namespace = "foo-namespace"
var name = "foo-deployment"

func validNewDeployment() *experimental.Deployment {
	return &experimental.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: experimental.DeploymentSpec{
			Selector: map[string]string{"a": "b"},
			Template: &api.PodTemplateSpec{
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
		Status: experimental.DeploymentStatus{
			Replicas: 5,
		},
	}
}

var validDeployment = *validNewDeployment()

func validNewScale() *experimental.Scale {
	return &experimental.Scale{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespace},
		Spec: experimental.ScaleSpec{
			Replicas: validDeployment.Spec.Replicas,
		},
		Status: experimental.ScaleStatus{
			Replicas: validDeployment.Status.Replicas,
			Selector: validDeployment.Spec.Template.Labels,
		},
	}
}

var validScale = *validNewScale()

func TestCreate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Deployment.Etcd)
	deployment := validNewDeployment()
	deployment.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		deployment,
		// invalid (invalid selector)
		&experimental.Deployment{
			Spec: experimental.DeploymentSpec{
				Selector: map[string]string{},
				Template: validDeployment.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Deployment.Etcd)
	test.TestUpdate(
		// valid
		validNewDeployment(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*experimental.Deployment)
			object.Spec.Template.Spec.NodeSelector = map[string]string{"c": "d"}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*experimental.Deployment)
			object.UID = "newUID"
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*experimental.Deployment)
			object.Name = ""
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*experimental.Deployment)
			object.Spec.Template.Spec.RestartPolicy = api.RestartPolicyOnFailure
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*experimental.Deployment)
			object.Spec.Selector = map[string]string{}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Deployment.Etcd)
	test.TestDelete(validNewDeployment())
}

func TestGet(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Deployment.Etcd)
	test.TestGet(validNewDeployment())
}

func TestList(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Deployment.Etcd)
	test.TestList(validNewDeployment())
}

func TestWatch(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Deployment.Etcd)
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
	storage, fakeClient := newStorage(t)

	ctx := api.WithNamespace(api.NewContext(), namespace)
	key := etcdtest.AddPrefix("/deployments/" + namespace + "/" + name)
	if _, err := fakeClient.Set(key, runtime.EncodeOrDie(testapi.Experimental.Codec(), &validDeployment), 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expect := &validScale
	obj, err := storage.Scale.Get(ctx, name)
	scale := obj.(*experimental.Scale)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e, a := expect, scale; !api.Semantic.DeepEqual(e, a) {
		t.Errorf("unexpected scale: %s", util.ObjectDiff(e, a))
	}
}

func TestScaleUpdate(t *testing.T) {
	storage, fakeClient := newStorage(t)

	ctx := api.WithNamespace(api.NewContext(), namespace)
	key := etcdtest.AddPrefix("/deployments/" + namespace + "/" + name)
	if _, err := fakeClient.Set(key, runtime.EncodeOrDie(testapi.Experimental.Codec(), &validDeployment), 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	replicas := 12
	update := experimental.Scale{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespace},
		Spec: experimental.ScaleSpec{
			Replicas: replicas,
		},
	}

	if _, _, err := storage.Scale.Update(ctx, &update); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var deployment experimental.Deployment
	testapi.Experimental.Codec().DecodeInto([]byte(response.Node.Value), &deployment)
	if deployment.Spec.Replicas != replicas {
		t.Errorf("wrong replicas count expected: %d got: %d", replicas, deployment.Spec.Replicas)
	}
}
