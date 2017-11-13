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

package storage

import (
	"testing"

	"k8s.io/api/batch/v2alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

// TODO: allow for global factory override
func newStorage(t *testing.T) (*REST, *StatusREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, batch.GroupName)
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1}
	storage, statusStorage := NewREST(restOptions)
	return storage, statusStorage, server
}

func validNewCronJob() *batch.CronJob {
	return &batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: batch.CronJobSpec{
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
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	validCronJob := validNewCronJob()
	validCronJob.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		validCronJob,
		// invalid (empty spec)
		&batch.CronJob{
			Spec: batch.CronJobSpec{},
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
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	schedule := "1 1 1 1 ?"
	test.TestUpdate(
		// valid
		validNewCronJob(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*batch.CronJob)
			object.Spec.Schedule = schedule
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*batch.CronJob)
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
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	test.TestDelete(validNewCronJob())
}

func TestGet(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	test.TestGet(validNewCronJob())
}

func TestList(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	test.TestList(validNewCronJob())
}

func TestWatch(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store, legacyscheme.Scheme)
	test.TestWatch(
		validNewCronJob(),
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
