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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	// Ensure that extensions/v1beta1 package is initialized.
	_ "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, extensions.GroupName)
	restOptions := generic.RESTOptions{etcdStorage, generic.UndecoratedStorage, 1}
	jobStorage, statusStorage := NewREST(restOptions)
	return jobStorage, statusStorage, server
}

func validNewJob() *extensions.Job {
	completions := 1
	parallelism := 1
	return &extensions.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
		},
		Spec: extensions.JobSpec{
			Completions: &completions,
			Parallelism: &parallelism,
			Selector: &unversioned.LabelSelector{
				MatchLabels: map[string]string{"a": "b"},
			},
			ManualSelector: newBool(true),
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
					RestartPolicy: api.RestartPolicyOnFailure,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	validJob := validNewJob()
	validJob.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		validJob,
		// invalid (empty selector)
		&extensions.Job{
			Spec: extensions.JobSpec{
				Completions: validJob.Spec.Completions,
				Selector:    &unversioned.LabelSelector{},
				Template:    validJob.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	two := 2
	test.TestUpdate(
		// valid
		validNewJob(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Job)
			object.Spec.Parallelism = &two
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Job)
			object.Spec.Selector = &unversioned.LabelSelector{}
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Job)
			object.Spec.Completions = &two
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestDelete(validNewJob())
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestGet(validNewJob())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestList(validNewJob())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
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
