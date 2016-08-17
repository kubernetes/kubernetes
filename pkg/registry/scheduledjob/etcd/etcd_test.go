/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, batch.GroupName)
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1}
	storage, statusStorage := NewREST(restOptions)
	return storage, statusStorage, server
}

func validNewScheduledJob() *batch.ScheduledJob {
	return &batch.ScheduledJob{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: batch.ScheduledJobSpec{
			Schedule:          "* * * * ?",
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				Spec: batch.JobSpec{
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{
							RestartPolicy: api.RestartPolicyOnFailure,
							DNSPolicy:     api.DNSClusterFirst,
							Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: api.PullIfNotPresent}},
						},
					},
				},
			},
		},
	}
}

func TestCreate(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	validScheduledJob := validNewScheduledJob()
	validScheduledJob.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		validScheduledJob,
		// invalid (empty spec)
		&batch.ScheduledJob{
			Spec: batch.ScheduledJobSpec{},
		},
	)
}

func TestUpdate(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	schedule := "1 1 1 1 ?"
	test.TestUpdate(
		// valid
		validNewScheduledJob(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*batch.ScheduledJob)
			object.Spec.Schedule = schedule
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*batch.ScheduledJob)
			object.Spec.Schedule = "* * *"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestDelete(validNewScheduledJob())
}

func TestGet(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestGet(validNewScheduledJob())
}

func TestList(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestList(validNewScheduledJob())
}

func TestWatch(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store)
	test.TestWatch(
		validNewScheduledJob(),
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
