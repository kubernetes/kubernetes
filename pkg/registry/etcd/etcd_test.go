/*
Copyright 2014 Google Inc. All rights reserved.

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
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"

	"github.com/coreos/go-etcd/etcd"
)

func NewTestEtcdRegistry(client tools.EtcdClient) *Registry {
	registry := NewRegistry(tools.EtcdHelper{client, latest.Codec, tools.RuntimeVersionAdapter{latest.ResourceVersioner}},
		&pod.BasicBoundPodFactory{})
	return registry
}

// TestEtcdGetPodDifferentNamespace ensures same-name pods in different namespaces do not clash
func TestEtcdGetPodDifferentNamespace(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)

	ctx1 := api.NewDefaultContext()
	ctx2 := api.WithNamespace(api.NewContext(), "other")

	key1, _ := makePodKey(ctx1, "foo")
	key2, _ := makePodKey(ctx2, "foo")

	fakeClient.Set(key1, runtime.EncodeOrDie(latest.Codec, &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: "default", Name: "foo"}}), 0)
	fakeClient.Set(key2, runtime.EncodeOrDie(latest.Codec, &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: "other", Name: "foo"}}), 0)

	registry := NewTestEtcdRegistry(fakeClient)

	pod1, err := registry.GetPod(ctx1, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if pod1.Name != "foo" {
		t.Errorf("Unexpected pod: %#v", pod1)
	}
	if pod1.Namespace != "default" {
		t.Errorf("Unexpected pod: %#v", pod1)
	}

	pod2, err := registry.GetPod(ctx2, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if pod2.Name != "foo" {
		t.Errorf("Unexpected pod: %#v", pod2)
	}
	if pod2.Namespace != "other" {
		t.Errorf("Unexpected pod: %#v", pod2)
	}

}

func TestEtcdGetPod(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key, _ := makePodKey(ctx, "foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	pod, err := registry.GetPod(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.Name != "foo" {
		t.Errorf("Unexpected pod: %#v", pod)
	}
}

func TestEtcdGetPodNotFound(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key, _ := makePodKey(ctx, "foo")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.GetPod(ctx, "foo")
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdCreatePod(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	key, _ := makePodKey(ctx, "foo")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	fakeClient.Set("/registry/nodes/machine/boundpods", runtime.EncodeOrDie(latest.Codec, &api.BoundPods{}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreatePod(ctx, &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "foo",
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	err = registry.ApplyBinding(ctx, &api.Binding{PodID: "foo", Host: "machine", ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var pod api.Pod
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.Name != "foo" {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
	var boundPods api.BoundPods
	resp, err = fakeClient.Get("/registry/nodes/machine/boundpods", false, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &boundPods)
	if len(boundPods.Items) != 1 || boundPods.Items[0].Name != "foo" {
		t.Errorf("Unexpected boundPod list: %#v", boundPods)
	}
}

func TestEtcdCreatePodFailsWithoutNamespace(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreatePod(api.NewContext(), &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "foo",
				},
			},
		},
	})
	// Accept "namespace" or "Namespace".
	if err == nil || !strings.Contains(err.Error(), "amespace") {
		t.Fatalf("expected error that namespace was missing from context, got: %v", err)
	}
}

func TestEtcdCreatePodAlreadyExisting(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key, _ := makePodKey(ctx, "foo")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}),
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreatePod(ctx, &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	})
	if !errors.IsAlreadyExists(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdCreatePodWithContainersError(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	key, _ := makePodKey(ctx, "foo")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	fakeClient.Data["/registry/nodes/machine/boundpods"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNodeExist, // validate that ApplyBinding is translating Create errors
	}
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreatePod(ctx, &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
	})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	err = registry.ApplyBinding(ctx, &api.Binding{PodID: "foo", Host: "machine"})
	if !errors.IsAlreadyExists(err) {
		t.Fatalf("Unexpected error returned: %#v", err)
	}

	existingPod, err := registry.GetPod(ctx, "foo")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if existingPod.Status.Host == "machine" {
		t.Fatal("Pod's host changed in response to an non-apply-able binding.")
	}
}

func TestEtcdCreatePodWithContainersNotFound(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	key, _ := makePodKey(ctx, "foo")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	fakeClient.Data["/registry/nodes/machine/boundpods"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreatePod(ctx, &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "foo",
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	err = registry.ApplyBinding(ctx, &api.Binding{PodID: "foo", Host: "machine"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var pod api.Pod
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.Name != "foo" {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
	var boundPods api.BoundPods
	resp, err = fakeClient.Get("/registry/nodes/machine/boundpods", false, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &boundPods)
	if len(boundPods.Items) != 1 || boundPods.Items[0].Name != "foo" {
		t.Errorf("Unexpected boundPod list: %#v", boundPods)
	}
}

func TestEtcdCreatePodWithExistingContainers(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	key, _ := makePodKey(ctx, "foo")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	fakeClient.Set("/registry/nodes/machine/boundpods", runtime.EncodeOrDie(latest.Codec, &api.BoundPods{
		Items: []api.BoundPod{
			{ObjectMeta: api.ObjectMeta{Name: "bar"}},
		},
	}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreatePod(ctx, &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "foo",
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	err = registry.ApplyBinding(ctx, &api.Binding{PodID: "foo", Host: "machine"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var pod api.Pod
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.Name != "foo" {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
	var boundPods api.BoundPods
	resp, err = fakeClient.Get("/registry/nodes/machine/boundpods", false, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &boundPods)
	if len(boundPods.Items) != 2 || boundPods.Items[1].Name != "foo" {
		t.Errorf("Unexpected boundPod list: %#v", boundPods)
	}
}

func TestEtcdUpdatePodNotFound(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true

	key, _ := makePodKey(ctx, "foo")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}

	registry := NewTestEtcdRegistry(fakeClient)
	podIn := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
	}
	err := registry.UpdatePod(ctx, &podIn)
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestEtcdUpdatePodNotScheduled(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true

	key, _ := makePodKey(ctx, "foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}), 1)

	registry := NewTestEtcdRegistry(fakeClient)
	podIn := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
			DNSPolicy:     api.DNSClusterFirst,
		},
	}
	err := registry.UpdatePod(ctx, &podIn)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	var podOut api.Pod
	latest.Codec.DecodeInto([]byte(response.Node.Value), &podOut)
	if !api.Semantic.DeepEqual(podOut, podIn) {
		t.Errorf("expected: %v, got: %v", podOut, podIn)
	}
}

func TestEtcdUpdatePodScheduled(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true

	key, _ := makePodKey(ctx, "foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.PodSpec{
			//			Host: "machine",
			Containers: []api.Container{
				{
					Image: "foo:v1",
				},
			},
		},
		Status: api.PodStatus{
			Host: "machine",
		},
	}), 1)

	contKey := "/registry/nodes/machine/boundpods"
	fakeClient.Set(contKey, runtime.EncodeOrDie(latest.Codec, &api.BoundPods{
		Items: []api.BoundPod{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:v1",
						},
					},
				},
			}, {
				ObjectMeta: api.ObjectMeta{Name: "bar"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:v1",
						},
					},
				},
			},
		},
	}), 0)

	registry := NewTestEtcdRegistry(fakeClient)
	podIn := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Image:                  "foo:v2",
					ImagePullPolicy:        api.PullIfNotPresent,
					TerminationMessagePath: api.TerminationMessagePathDefault,
				},
			},
			RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
			DNSPolicy:     api.DNSClusterFirst,
		},
		Status: api.PodStatus{
			Host: "machine",
		},
	}
	err := registry.UpdatePod(ctx, &podIn)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	var podOut api.Pod
	latest.Codec.DecodeInto([]byte(response.Node.Value), &podOut)
	if !api.Semantic.DeepEqual(podOut, podIn) {
		t.Errorf("expected: %#v, got: %#v", podOut, podIn)
	}

	response, err = fakeClient.Get(contKey, false, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	var list api.BoundPods
	if err := latest.Codec.DecodeInto([]byte(response.Node.Value), &list); err != nil {
		t.Fatalf("unexpected error decoding response: %v", err)
	}

	if len(list.Items) != 2 || !api.Semantic.DeepEqual(list.Items[0].Spec, podIn.Spec) {
		t.Errorf("unexpected container list: %d\n items[0] -   %#v\n podin.spec - %#v\n", len(list.Items), list.Items[0].Spec, podIn.Spec)
	}
}

func TestEtcdDeletePod(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true

	key, _ := makePodKey(ctx, "foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Status:     api.PodStatus{Host: "machine"},
	}), 0)
	fakeClient.Set("/registry/nodes/machine/boundpods", runtime.EncodeOrDie(latest.Codec, &api.BoundPods{
		Items: []api.BoundPod{
			{ObjectMeta: api.ObjectMeta{Name: "foo"}},
		},
	}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.DeletePod(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	} else if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
	response, err := fakeClient.Get("/registry/nodes/machine/boundpods", false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var boundPods api.BoundPods
	latest.Codec.DecodeInto([]byte(response.Node.Value), &boundPods)
	if len(boundPods.Items) != 0 {
		t.Errorf("Unexpected container set: %s, expected empty", response.Node.Value)
	}
}

func TestEtcdDeletePodMultipleContainers(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	key, _ := makePodKey(ctx, "foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Status:     api.PodStatus{Host: "machine"},
	}), 0)
	fakeClient.Set("/registry/nodes/machine/boundpods", runtime.EncodeOrDie(latest.Codec, &api.BoundPods{
		Items: []api.BoundPod{
			{ObjectMeta: api.ObjectMeta{Name: "foo"}},
			{ObjectMeta: api.ObjectMeta{Name: "bar"}},
		},
	}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.DeletePod(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	}
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
	response, err := fakeClient.Get("/registry/nodes/machine/boundpods", false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var boundPods api.BoundPods
	latest.Codec.DecodeInto([]byte(response.Node.Value), &boundPods)
	if len(boundPods.Items) != 1 {
		t.Fatalf("Unexpected boundPod set: %#v, expected empty", boundPods)
	}
	if boundPods.Items[0].Name != "bar" {
		t.Errorf("Deleted wrong boundPod: %#v", boundPods)
	}
}

func TestEtcdEmptyListPods(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ctx := api.NewDefaultContext()
	key := makePodListKey(ctx)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{},
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	pods, err := registry.ListPods(ctx, labels.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(pods.Items) != 0 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestEtcdListPodsNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ctx := api.NewDefaultContext()
	key := makePodListKey(ctx)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	pods, err := registry.ListPods(ctx, labels.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(pods.Items) != 0 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestEtcdListPods(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ctx := api.NewDefaultContext()
	key := makePodListKey(ctx)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Pod{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
							Status:     api.PodStatus{Host: "machine"},
						}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Pod{
							ObjectMeta: api.ObjectMeta{Name: "bar"},
							Status:     api.PodStatus{Host: "machine"},
						}),
					},
				},
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	pods, err := registry.ListPods(ctx, labels.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(pods.Items) != 2 || pods.Items[0].Name != "foo" || pods.Items[1].Name != "bar" {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
	if pods.Items[0].Status.Host != "machine" ||
		pods.Items[1].Status.Host != "machine" {
		t.Errorf("Failed to populate host name.")
	}
}

func TestEtcdWatchPods(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchPods(ctx,
		labels.Everything(),
		labels.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	default:
	}
	fakeClient.WatchInjectError <- nil
	if _, ok := <-watching.ResultChan(); ok {
		t.Errorf("watching channel should be closed")
	}
	watching.Stop()
}

func TestEtcdWatchPodsMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchPods(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		labels.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				"name": "foo",
			},
		},
	}
	podBytes, _ := latest.Codec.Encode(pod)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(podBytes),
		},
	}
	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	case <-time.After(time.Millisecond * 100):
		t.Error("unexpected timeout from result channel")
	}
	watching.Stop()
}

func TestEtcdWatchPodsNotMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchPods(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		labels.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "bar",
			Labels: map[string]string{
				"name": "bar",
			},
		},
	}
	podBytes, _ := latest.Codec.Encode(pod)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(podBytes),
		},
	}

	select {
	case <-watching.ResultChan():
		t.Error("unexpected result from result channel")
	case <-time.After(time.Millisecond * 100):
		// expected case
	}
}

func TestEtcdListControllersNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ctx := api.NewDefaultContext()
	key := makeControllerListKey(ctx)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	controllers, err := registry.ListControllers(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(controllers.Items) != 0 {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

func TestEtcdListServicesNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ctx := api.NewDefaultContext()
	key := makeServiceListKey(ctx)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	services, err := registry.ListServices(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(services.Items) != 0 {
		t.Errorf("Unexpected controller list: %#v", services)
	}
}

func TestEtcdListControllers(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ctx := api.NewDefaultContext()
	key := makeControllerListKey(ctx)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "bar"}}),
					},
				},
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	controllers, err := registry.ListControllers(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(controllers.Items) != 2 || controllers.Items[0].Name != "foo" || controllers.Items[1].Name != "bar" {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

// TestEtcdGetControllerDifferentNamespace ensures same-name controllers in different namespaces do not clash
func TestEtcdGetControllerDifferentNamespace(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)

	ctx1 := api.NewDefaultContext()
	ctx2 := api.WithNamespace(api.NewContext(), "other")

	key1, _ := makeControllerKey(ctx1, "foo")
	key2, _ := makeControllerKey(ctx2, "foo")

	fakeClient.Set(key1, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Namespace: "default", Name: "foo"}}), 0)
	fakeClient.Set(key2, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Namespace: "other", Name: "foo"}}), 0)

	registry := NewTestEtcdRegistry(fakeClient)

	ctrl1, err := registry.GetController(ctx1, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if ctrl1.Name != "foo" {
		t.Errorf("Unexpected controller: %#v", ctrl1)
	}
	if ctrl1.Namespace != "default" {
		t.Errorf("Unexpected controller: %#v", ctrl1)
	}

	ctrl2, err := registry.GetController(ctx2, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if ctrl2.Name != "foo" {
		t.Errorf("Unexpected controller: %#v", ctrl2)
	}
	if ctrl2.Namespace != "other" {
		t.Errorf("Unexpected controller: %#v", ctrl2)
	}

}

func TestEtcdGetController(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key, _ := makeControllerKey(ctx, "foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	ctrl, err := registry.GetController(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if ctrl.Name != "foo" {
		t.Errorf("Unexpected controller: %#v", ctrl)
	}
}

func TestEtcdGetControllerNotFound(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key, _ := makeControllerKey(ctx, "foo")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	ctrl, err := registry.GetController(ctx, "foo")
	if ctrl != nil {
		t.Errorf("Unexpected non-nil controller: %#v", ctrl)
	}
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdDeleteController(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	key, _ := makeControllerKey(ctx, "foo")
	err := registry.DeleteController(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	}
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
}

func TestEtcdCreateController(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	key, _ := makeControllerKey(ctx, "foo")
	err := registry.CreateController(ctx, &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var ctrl api.ReplicationController
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &ctrl)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if ctrl.Name != "foo" {
		t.Errorf("Unexpected pod: %#v %s", ctrl, resp.Node.Value)
	}
}

func TestEtcdCreateControllerAlreadyExisting(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key, _ := makeControllerKey(ctx, "foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)

	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreateController(ctx, &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	})
	if !errors.IsAlreadyExists(err) {
		t.Errorf("expected already exists err, got %#v", err)
	}
}

func TestEtcdUpdateController(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	key, _ := makeControllerKey(ctx, "foo")
	resp, _ := fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.UpdateController(ctx, &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: strconv.FormatUint(resp.Node.ModifiedIndex, 10)},
		Spec: api.ReplicationControllerSpec{
			Replicas: 2,
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	ctrl, err := registry.GetController(ctx, "foo")
	if ctrl.Spec.Replicas != 2 {
		t.Errorf("Unexpected controller: %#v", ctrl)
	}
}

func TestEtcdWatchController(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchControllers(ctx,
		labels.Everything(),
		labels.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	default:
	}
	fakeClient.WatchInjectError <- nil
	if _, ok := <-watching.ResultChan(); ok {
		t.Errorf("watching channel should be closed")
	}
	watching.Stop()
}

func TestEtcdWatchControllersMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.ExpectNotFoundGet(makePodListKey(ctx))
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchControllers(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		labels.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				"name": "foo",
			},
		},
	}
	controllerBytes, _ := latest.Codec.Encode(controller)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(controllerBytes),
		},
	}
	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	case <-time.After(time.Millisecond * 100):
		t.Error("unexpected timeout from result channel")
	}
	watching.Stop()
}

func TestEtcdWatchControllersNotMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.ExpectNotFoundGet(makePodListKey(ctx))
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchControllers(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		labels.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "bar",
			Labels: map[string]string{
				"name": "bar",
			},
		},
	}
	controllerBytes, _ := latest.Codec.Encode(controller)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(controllerBytes),
		},
	}

	select {
	case <-watching.ResultChan():
		t.Error("unexpected result from result channel")
	case <-time.After(time.Millisecond * 100):
		// expected case
	}
}

func TestEtcdListServices(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key := makeServiceListKey(ctx)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Name: "foo"}}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Name: "bar"}}),
					},
				},
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	services, err := registry.ListServices(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(services.Items) != 2 || services.Items[0].Name != "foo" || services.Items[1].Name != "bar" {
		t.Errorf("Unexpected service list: %#v", services)
	}
}

func TestEtcdCreateService(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	key, _ := makeServiceKey(ctx, "foo")
	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	var service api.Service
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &service)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if service.Name != "foo" {
		t.Errorf("Unexpected service: %#v %s", service, resp.Node.Value)
	}
}

func TestEtcdCreateServiceAlreadyExisting(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key, _ := makeServiceKey(ctx, "foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	})
	if !errors.IsAlreadyExists(err) {
		t.Errorf("expected already exists err, got %#v", err)
	}
}

// TestEtcdGetServiceDifferentNamespace ensures same-name services in different namespaces do not clash
func TestEtcdGetServiceDifferentNamespace(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)

	ctx1 := api.NewDefaultContext()
	ctx2 := api.WithNamespace(api.NewContext(), "other")

	key1, _ := makeServiceKey(ctx1, "foo")
	key2, _ := makeServiceKey(ctx2, "foo")

	fakeClient.Set(key1, runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Namespace: "default", Name: "foo"}}), 0)
	fakeClient.Set(key2, runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Namespace: "other", Name: "foo"}}), 0)

	registry := NewTestEtcdRegistry(fakeClient)

	service1, err := registry.GetService(ctx1, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if service1.Name != "foo" {
		t.Errorf("Unexpected service: %#v", service1)
	}
	if service1.Namespace != "default" {
		t.Errorf("Unexpected service: %#v", service1)
	}

	service2, err := registry.GetService(ctx2, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if service2.Name != "foo" {
		t.Errorf("Unexpected service: %#v", service2)
	}
	if service2.Namespace != "other" {
		t.Errorf("Unexpected service: %#v", service2)
	}

}

func TestEtcdGetService(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key, _ := makeServiceKey(ctx, "foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	service, err := registry.GetService(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if service.Name != "foo" {
		t.Errorf("Unexpected service: %#v", service)
	}
}

func TestEtcdGetServiceNotFound(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key, _ := makeServiceKey(ctx, "foo")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.GetService(ctx, "foo")
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdDeleteService(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.DeleteService(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 2 {
		t.Errorf("Expected 2 delete, found %#v", fakeClient.DeletedKeys)
	}
	key, _ := makeServiceKey(ctx, "foo")
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
	key, _ = makeServiceEndpointsKey(ctx, "foo")
	if fakeClient.DeletedKeys[1] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[1], key)
	}
}

func TestEtcdUpdateService(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	key, _ := makeServiceKey(ctx, "uniquefoo")
	resp, _ := fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Name: "uniquefoo"}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	testService := api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:            "uniquefoo",
			ResourceVersion: strconv.FormatUint(resp.Node.ModifiedIndex, 10),
			Labels: map[string]string{
				"baz": "bar",
			},
		},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"baz": "bar",
			},
			Protocol:        "TCP",
			SessionAffinity: "None",
		},
	}
	err := registry.UpdateService(ctx, &testService)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	svc, err := registry.GetService(ctx, "uniquefoo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Clear modified indices before the equality test.
	svc.ResourceVersion = ""
	testService.ResourceVersion = ""
	if !api.Semantic.DeepEqual(*svc, testService) {
		t.Errorf("Unexpected service: got\n %#v\n, wanted\n %#v", svc, testService)
	}
}

func TestEtcdListEndpoints(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key := makeServiceEndpointsListKey(ctx)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Endpoints{ObjectMeta: api.ObjectMeta{Name: "foo"}, Endpoints: []string{"127.0.0.1:8345"}}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Endpoints{ObjectMeta: api.ObjectMeta{Name: "bar"}}),
					},
				},
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	services, err := registry.ListEndpoints(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(services.Items) != 2 || services.Items[0].Name != "foo" || services.Items[1].Name != "bar" {
		t.Errorf("Unexpected endpoints list: %#v", services)
	}
}

func TestEtcdGetEndpoints(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	endpoints := &api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Endpoints:  []string{"127.0.0.1:34855"},
	}

	key, _ := makeServiceEndpointsKey(ctx, "foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, endpoints), 0)

	got, err := registry.GetEndpoints(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if e, a := endpoints, got; !api.Semantic.DeepEqual(e, a) {
		t.Errorf("Unexpected endpoints: %#v, expected %#v", e, a)
	}
}

func TestEtcdUpdateEndpoints(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	registry := NewTestEtcdRegistry(fakeClient)
	endpoints := api.Endpoints{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Endpoints:  []string{"baz", "bar"},
	}

	key, _ := makeServiceEndpointsKey(ctx, "foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Endpoints{}), 0)

	err := registry.UpdateEndpoints(ctx, &endpoints)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var endpointsOut api.Endpoints
	err = latest.Codec.DecodeInto([]byte(response.Node.Value), &endpointsOut)
	if !api.Semantic.DeepEqual(endpoints, endpointsOut) {
		t.Errorf("Unexpected endpoints: %#v, expected %#v", endpointsOut, endpoints)
	}
}

func TestEtcdWatchServices(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchServices(ctx,
		labels.Everything(),
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	default:
	}
	fakeClient.WatchInjectError <- nil
	if _, ok := <-watching.ResultChan(); ok {
		t.Errorf("watching channel should be closed")
	}
	watching.Stop()
}

func TestEtcdWatchServicesBadSelector(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.WatchServices(
		ctx,
		labels.Everything(),
		labels.SelectorFromSet(labels.Set{"Field.Selector": "foo"}),
		"",
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}

	_, err = registry.WatchServices(
		ctx,
		labels.SelectorFromSet(labels.Set{"Label.Selector": "foo"}),
		labels.Everything(),
		"",
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}
}

func TestEtcdWatchEndpoints(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchEndpoints(
		ctx,
		labels.Everything(),
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	default:
	}
	fakeClient.WatchInjectError <- nil
	if _, ok := <-watching.ResultChan(); ok {
		t.Errorf("watching channel should be closed")
	}
	watching.Stop()
}

func TestEtcdWatchEndpointsAcrossNamespaces(t *testing.T) {
	ctx := api.NewContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchEndpoints(
		ctx,
		labels.Everything(),
		labels.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	default:
	}
	fakeClient.WatchInjectError <- nil
	if _, ok := <-watching.ResultChan(); ok {
		t.Errorf("watching channel should be closed")
	}
	watching.Stop()
}

func TestEtcdWatchEndpointsBadSelector(t *testing.T) {
	ctx := api.NewContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.WatchEndpoints(
		ctx,
		labels.Everything(),
		labels.SelectorFromSet(labels.Set{"Field.Selector": "foo"}),
		"",
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}

	_, err = registry.WatchEndpoints(
		ctx,
		labels.SelectorFromSet(labels.Set{"Label.Selector": "foo"}),
		labels.Everything(),
		"",
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}
}

func TestEtcdListMinions(t *testing.T) {
	ctx := api.NewContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	key := "/registry/minions"
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Node{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
						}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Node{
							ObjectMeta: api.ObjectMeta{Name: "bar"},
						}),
					},
				},
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	minions, err := registry.ListMinions(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(minions.Items) != 2 || minions.Items[0].Name != "foo" || minions.Items[1].Name != "bar" {
		t.Errorf("Unexpected minion list: %#v", minions)
	}
}

func TestEtcdCreateMinion(t *testing.T) {
	ctx := api.NewContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreateMinion(ctx, &api.Node{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	resp, err := fakeClient.Get("/registry/minions/foo", false, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	var minion api.Node
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &minion)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if minion.Name != "foo" {
		t.Errorf("Unexpected minion: %#v %s", minion, resp.Node.Value)
	}
}

func TestEtcdGetMinion(t *testing.T) {
	ctx := api.NewContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Set("/registry/minions/foo", runtime.EncodeOrDie(latest.Codec, &api.Node{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	minion, err := registry.GetMinion(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if minion.Name != "foo" {
		t.Errorf("Unexpected minion: %#v", minion)
	}
}

func TestEtcdGetMinionNotFound(t *testing.T) {
	ctx := api.NewContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Data["/registry/minions/foo"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.GetMinion(ctx, "foo")

	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdDeleteMinion(t *testing.T) {
	ctx := api.NewContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.DeleteMinion(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	}
	key := "/registry/minions/foo"
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
}

func TestEtcdWatchMinion(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchMinions(ctx,
		labels.Everything(),
		labels.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	default:
	}
	fakeClient.WatchInjectError <- nil
	if _, ok := <-watching.ResultChan(); ok {
		t.Errorf("watching channel should be closed")
	}
	watching.Stop()
}

func TestEtcdWatchMinionsMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchMinions(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		labels.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	node := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				"name": "foo",
			},
		},
	}
	nodeBytes, _ := latest.Codec.Encode(node)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(nodeBytes),
		},
	}
	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	case <-time.After(time.Millisecond * 100):
		t.Error("unexpected timeout from result channel")
	}
	watching.Stop()
}

func TestEtcdWatchMinionsNotMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchMinions(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		labels.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	node := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: "bar",
			Labels: map[string]string{
				"name": "bar",
			},
		},
	}
	nodeBytes, _ := latest.Codec.Encode(node)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(nodeBytes),
		},
	}

	select {
	case <-watching.ResultChan():
		t.Error("unexpected result from result channel")
	case <-time.After(time.Millisecond * 100):
		// expected case
	}
}

// TODO We need a test for the compare and swap behavior.  This basically requires two things:
//   1) Add a per-operation synchronization channel to the fake etcd client, such that any operation waits on that
//      channel, this will enable us to orchestrate the flow of etcd requests in the test.
//   2) We need to make the map from key to (response, error) actually be a [](response, error) and pop
//      our way through the responses.  That will enable us to hand back multiple different responses for
//      the same key.
//   Once that infrastructure is in place, the test looks something like:
//      Routine #1                               Routine #2
//         Read
//         Wait for sync on update               Read
//                                               Update
//         Update
//   In the buggy case, this will result in lost data.  In the correct case, the second update should fail
//   and be retried.
