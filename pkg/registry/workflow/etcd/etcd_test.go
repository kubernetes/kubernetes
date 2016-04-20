/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/util/diff"
)

func newWorkflow(name string) *extensions.Workflow {
	return &extensions.Workflow{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.WorkflowSpec{
			Steps: map[string]extensions.WorkflowStep{
				"one": {},
			},
			Selector: &unversioned.LabelSelector{
				MatchLabels: map[string]string{"a": "b"},
			},
		},
	}
}

func newStorage(t *testing.T) (*REST, *StatusREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, batch.GroupName)
	restOptions := generic.RESTOptions{etcdStorage, generic.UndecoratedStorage, 1}
	workflowStorage, statusStorage := NewREST(restOptions)
	return workflowStorage, statusStorage, server
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	validWorkflow := newWorkflow("mydag")
	validWorkflow.ObjectMeta = api.ObjectMeta{}

	invalidWorkflow := newWorkflow("mydag")
	invalidWorkflow.ObjectMeta = api.ObjectMeta{}
	invalidWorkflow.Spec.Selector = &unversioned.LabelSelector{}
	test.TestCreate(validWorkflow, invalidWorkflow)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestUpdate(
		// valid
		newWorkflow("mydag"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Workflow)
			object.Spec.Steps["two"] = extensions.WorkflowStep{}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Workflow)
			object.Spec.Selector = &unversioned.LabelSelector{}
			return object
		},
		/* // TODO: fix it @sdminonne
		func(obj runtime.Object) runtime.Object {
					object := obj.(*extensions.Workflow)
					return object
				},
		*/
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestDelete(newWorkflow("mydag"))
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestGet(newWorkflow("mydag"))
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestList(newWorkflow("mydag"))
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Etcd)
	test.TestWatch(
		newWorkflow("mydag"),
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

func TestEtcdUpdateStatus(t *testing.T) {
	storage, status, server := newStorage(t)
	defer server.Terminate(t)
	ctx := api.NewDefaultContext()

	initialWorkflow := newWorkflow("foo")
	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)

	err := storage.Storage.Create(ctx, key, initialWorkflow, nil, 0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	in := newWorkflow("foo")
	in.Status = extensions.WorkflowStatus{
		StartTime: &unversioned.Time{time.Date(2009, time.January, 1, 27, 6, 25, 0, time.UTC)},
		Statuses: map[string]extensions.WorkflowStepStatus{
			"one": {
				Complete: true,
			},
		},
	}
	_, _, err = status.Update(ctx, in)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	obj, err := storage.Get(ctx, "foo")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	out := obj.(*extensions.Workflow)
	if !api.Semantic.DeepEqual(out.Status, in.Status) {
		t.Errorf("objects differ: %v", diff.ObjectDiff(out, in))
	}
}
