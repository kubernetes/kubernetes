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
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"

	"k8s.io/kubernetes/pkg/expapi"

	"github.com/coreos/go-etcd/etcd"
)

func newEtcdStorage(t *testing.T) (*tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeEtcdClient, latest.Codec, etcdtest.PathPrefix())
	return fakeEtcdClient, etcdStorage
}

func newStorage(t *testing.T) (*RcREST, *ScaleREST, *tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage)
	return storage.ReplicationController, storage.Scale, fakeEtcdClient, etcdStorage
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

var validScale = expapi.Scale{
	ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test"},
	Spec: expapi.ScaleSpec{
		Replicas: validReplicas,
	},
	Status: expapi.ScaleStatus{
		Replicas: 0,
		Selector: validPodTemplate.Template.Labels,
	},
}

func TestGet(t *testing.T) {
	expect := &validScale

	fakeEtcdClient, etcdStorage := newEtcdStorage(t)

	key := etcdtest.AddPrefix("/controllers/test/foo")
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(latest.Codec, &validController),
				ModifiedIndex: 1,
			},
		},
	}
	storage := NewStorage(etcdStorage).Scale

	obj, err := storage.Get(api.WithNamespace(api.NewContext(), "test"), "foo")
	scaler := obj.(*expapi.Scale)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e, a := expect, scaler; !api.Semantic.DeepEqual(e, a) {
		t.Errorf("Unexpected scaler: %s", util.ObjectDiff(e, a))
	}
}

func TestUpdate(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage := NewStorage(etcdStorage).Scale

	key := etcdtest.AddPrefix("/controllers/test/foo")
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(latest.Codec, &validController),
				ModifiedIndex: 1,
			},
		},
	}
	replicas := 12
	update := expapi.Scale{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec: expapi.ScaleSpec{
			Replicas: replicas,
		},
	}

	_, _, err := storage.Update(api.WithNamespace(api.NewContext(), "test"), &update)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	response, err := fakeEtcdClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	var controller api.ReplicationController
	latest.Codec.DecodeInto([]byte(response.Node.Value), &controller)
	if controller.Spec.Replicas != replicas {
		t.Errorf("wrong replicas count expected: %d got: %d", replicas, controller.Spec.Replicas)
	}
}
