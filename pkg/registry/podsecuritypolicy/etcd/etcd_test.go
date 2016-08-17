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

package etcd

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
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

func newStorage(t *testing.T) (*REST, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "extensions")
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1}
	return NewREST(restOptions), server
}

func validNewPodSecurityPolicy() *extensions.PodSecurityPolicy {
	return &extensions.PodSecurityPolicy{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
		Spec: extensions.PodSecurityPolicySpec{
			SELinux: extensions.SELinuxStrategyOptions{
				Rule: extensions.SELinuxStrategyRunAsAny,
			},
			RunAsUser: extensions.RunAsUserStrategyOptions{
				Rule: extensions.RunAsUserStrategyRunAsAny,
			},
			FSGroup: extensions.FSGroupStrategyOptions{
				Rule: extensions.FSGroupStrategyRunAsAny,
			},
			SupplementalGroups: extensions.SupplementalGroupsStrategyOptions{
				Rule: extensions.SupplementalGroupsStrategyRunAsAny,
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope()
	psp := validNewPodSecurityPolicy()
	psp.ObjectMeta = api.ObjectMeta{GenerateName: "foo-"}
	test.TestCreate(
		// valid
		psp,
		// invalid
		&extensions.PodSecurityPolicy{
			ObjectMeta: api.ObjectMeta{Name: "name with spaces"},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestUpdate(
		// valid
		validNewPodSecurityPolicy(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*extensions.PodSecurityPolicy)
			object.Labels = map[string]string{"a": "b"}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewPodSecurityPolicy())
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewPodSecurityPolicy())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewPodSecurityPolicy())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	test := registrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validNewPodSecurityPolicy(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
			{"name": "foo"},
		},
	)
}
