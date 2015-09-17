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
	"k8s.io/kubernetes/pkg/apis/experimental"
	// Ensure that experimental/v1alpha1 package is initialized.
	_ "k8s.io/kubernetes/pkg/apis/experimental/v1alpha1"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t, "experimental")
	storage, statusStorage := NewREST(etcdStorage)
	return storage, statusStorage, fakeClient
}

func validNewJob() *experimental.Job {
	completions := 1
	parallelism := 1
	return &experimental.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
		},
		Spec: experimental.JobSpec{
			Completions: &completions,
			Parallelism: &parallelism,
			Selector:    map[string]string{"a": "b"},
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
					RestartPolicy: api.RestartPolicyOnFailure,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd)
	validJob := validNewJob()
	validJob.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		validJob,
		// invalid (empty selector)
		&experimental.Job{
			Spec: experimental.JobSpec{
				Completions: validJob.Spec.Completions,
				Selector:    map[string]string{},
				Template:    validJob.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd)
	two := 2
	test.TestUpdate(
		// valid
		validNewJob(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*experimental.Job)
			object.Spec.Parallelism = &two
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*experimental.Job)
			object.Spec.Selector = map[string]string{}
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*experimental.Job)
			object.Spec.Completions = &two
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd)
	test.TestDelete(validNewJob())
}

func TestGet(t *testing.T) {
	storage, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd)
	test.TestGet(validNewJob())
}

func TestList(t *testing.T) {
	storage, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd)
	test.TestList(validNewJob())
}

func TestWatch(t *testing.T) {
	storage, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd)
	test.TestWatch(
		validNewJob(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"x": "y"},
		},
		// matching fields
		[]fields.Set{},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "xyz"},
			{"name": "foo"},
		},
	)
}

// TODO: test update /status
