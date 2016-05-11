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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
)

var namespace = "foo-namespace"
var name = "foo-template"

func newRest(t *testing.T) (*REST, *ProcessREST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, extensions.GroupName)
	restOptions := generic.RESTOptions{Storage: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1}
	rest, processRest := NewREST(restOptions)
	return rest, processRest, server
}

func validNewTemplate() *extensions.Template {
	return &extensions.Template{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: extensions.TemplateSpec{
			Objects:    []runtime.RawExtension{},
			Parameters: []extensions.Parameter{},
		},
	}
}

func TestCreate(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)
	template := validNewTemplate()
	template.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		template,
		// TODO: Add invalid cases once template Validation is implemented
	)
}

func TestUpdate(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)
	test.TestUpdate(
		// valid object
		validNewTemplate(),
		// valid update
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.Template)
			return object
		},
		// TODO: Add invalid cases once template Validation is implemented
	)
}

func TestDelete(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)
	test.TestDelete(validNewTemplate())
}

func TestGet(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)
	test.TestGet(validNewTemplate())
}

func TestList(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)
	test.TestList(validNewTemplate())
}

func TestWatch(t *testing.T) {
	rest, _, server := newRest(t)
	defer server.Terminate(t)
	test := registrytest.New(t, rest.Store)
	test.TestWatch(
		validNewTemplate(),
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
