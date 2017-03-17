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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*JobStorage, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, batch.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "jobs",
	}
	jobStorage := NewStorage(restOptions)
	return &jobStorage, server
}

func validNewJob() *batch.Job {
	completions := int32(1)
	parallelism := int32(1)
	return &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
		},
		Spec: batch.JobSpec{
			Completions: &completions,
			Parallelism: &parallelism,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"a": "b"},
			},
			ManualSelector: newBool(true),
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
					RestartPolicy: api.RestartPolicyOnFailure,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := registrytest.New(t, storage.Job.Store)
	validJob := validNewJob()
	validJob.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		validJob,
		// invalid (empty selector)
		&batch.Job{
			Spec: batch.JobSpec{
				Completions: validJob.Spec.Completions,
				Selector:    &metav1.LabelSelector{},
				Template:    validJob.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := registrytest.New(t, storage.Job.Store)
	two := int32(2)
	test.TestUpdate(
		// valid
		validNewJob(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*batch.Job)
			object.Spec.Parallelism = &two
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*batch.Job)
			object.Spec.Selector = &metav1.LabelSelector{}
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*batch.Job)
			object.Spec.Completions = &two
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := registrytest.New(t, storage.Job.Store)
	test.TestDelete(validNewJob())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := registrytest.New(t, storage.Job.Store)
	test.TestGet(validNewJob())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := registrytest.New(t, storage.Job.Store)
	test.TestList(validNewJob())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := registrytest.New(t, storage.Job.Store)
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

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}

func TestCategories(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	expected := []string{"all"}
	registrytest.AssertCategories(t, storage.Job, expected)
}
