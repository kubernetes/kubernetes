/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
)

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t)
	return NewREST(etcdStorage), fakeClient
}

func validNewPodTemplate(name string) *api.PodTemplate {
	return &api.PodTemplate{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: map[string]string{"test": "foo"},
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers: []api.Container{
					{
						Name:            "foo",
						Image:           "test",
						ImagePullPolicy: api.PullAlways,

						TerminationMessagePath: api.TerminationMessagePathDefault,
					},
				},
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	pod := validNewPodTemplate("foo")
	pod.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		pod,
		// invalid
		&api.PodTemplate{
			Template: api.PodTemplateSpec{},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	key, err := storage.KeyFunc(test.TestContext(), "foo")
	if err != nil {
		t.Fatal(err)
	}
	key = etcdtest.AddPrefix(key)

	fakeClient.ExpectNotFoundGet(key)
	fakeClient.ChangeIndex = 2
	pod := validNewPodTemplate("foo")
	existing := validNewPodTemplate("exists")
	existing.Namespace = test.TestNamespace()
	obj, err := storage.Create(test.TestContext(), existing)
	if err != nil {
		t.Fatalf("unable to create object: %v", err)
	}
	older := obj.(*api.PodTemplate)
	older.ResourceVersion = "1"

	test.TestUpdate(
		pod,
		existing,
		older,
	)
}
