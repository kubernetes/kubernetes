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
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"
)

func newStorage(t *testing.T) (*ScaleREST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t, "")
	return NewStorage(etcdStorage).Scale, fakeClient
}

var validPodTemplate = api.PodTemplate{
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
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
	},
}

var validReplicas = 8

var validControllerSpec = api.ReplicationControllerSpec{
	Replicas: validReplicas,
	Selector: validPodTemplate.Template.Labels,
	Template: &validPodTemplate.Template,
}

var validController = api.ReplicationController{
	ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "1"},
	Spec:       validControllerSpec,
}

var validScale = extensions.Scale{
	ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test"},
	Spec: extensions.ScaleSpec{
		Replicas: validReplicas,
	},
	Status: extensions.ScaleStatus{
		Replicas: 0,
		Selector: validPodTemplate.Template.Labels,
	},
}

func TestGet(t *testing.T) {
	storage, fakeClient := newStorage(t)

	ctx := api.WithNamespace(api.NewContext(), "test")
	key := etcdtest.AddPrefix("/controllers/test/foo")
	if _, err := fakeClient.Set(key, runtime.EncodeOrDie(testapi.Default.Codec(), &validController), 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expect := &validScale
	obj, err := storage.Get(ctx, "foo")
	scale := obj.(*extensions.Scale)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e, a := expect, scale; !api.Semantic.DeepEqual(e, a) {
		t.Errorf("unexpected scale: %s", util.ObjectDiff(e, a))
	}
}

func TestUpdate(t *testing.T) {
	storage, fakeClient := newStorage(t)

	ctx := api.WithNamespace(api.NewContext(), "test")
	key := etcdtest.AddPrefix("/controllers/test/foo")
	if _, err := fakeClient.Set(key, runtime.EncodeOrDie(testapi.Default.Codec(), &validController), 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	replicas := 12
	update := extensions.Scale{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec: extensions.ScaleSpec{
			Replicas: replicas,
		},
	}

	if _, _, err := storage.Update(ctx, &update); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var controller api.ReplicationController
	testapi.Extensions.Codec().DecodeInto([]byte(response.Node.Value), &controller)
	if controller.Spec.Replicas != replicas {
		t.Errorf("wrong replicas count expected: %d got: %d", replicas, controller.Spec.Replicas)
	}
}
