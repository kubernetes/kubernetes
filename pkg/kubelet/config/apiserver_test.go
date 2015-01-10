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

package config

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

const hostname string = "mcaa1"

type fakePodLW struct {
	listResp  runtime.Object
	watchResp watch.Interface
}

func (lw fakePodLW) List() (runtime.Object, error) {
	return lw.listResp, nil
}

func (lw fakePodLW) Watch(resourceVersion string) (watch.Interface, error) {
	return lw.watchResp, nil
}

var _ cache.ListerWatcher = fakePodLW{}

func TestNewSourceApiserver(t *testing.T) {
	podv1 := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "p"},
		Spec:       api.PodSpec{Containers: []api.Container{{Image: "image/one"}}}}
	podv2 := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "p"},
		Spec:       api.PodSpec{Containers: []api.Container{{Image: "image/two"}}}}

	expectedBoundPodv1 := api.BoundPod{
		ObjectMeta: api.ObjectMeta{Name: "p", SelfLink: "/api/v1beta1/boundPods/p"},
		Spec:       api.PodSpec{Containers: []api.Container{{Image: "image/one"}}}}
	expectedBoundPodv2 := api.BoundPod{
		ObjectMeta: api.ObjectMeta{Name: "p", SelfLink: "/api/v1beta1/boundPods/p"},
		Spec:       api.PodSpec{Containers: []api.Container{{Image: "image/two"}}}}

	// Setup fake api client.
	fakeWatch := watch.NewFake()
	lw := fakePodLW{
		listResp:  &api.PodList{Items: []api.Pod{podv1}},
		watchResp: fakeWatch,
	}

	ch := make(chan interface{})

	newSourceApiserverFromLW(lw, ch)

	got, ok := <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	update := got.(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET, kubelet.ApiserverSource, expectedBoundPodv1)
	if !api.Semantic.DeepEqual(expected, update) {
		t.Errorf("Expected %#v; Got %#v", expected, update)
	}

	fakeWatch.Modify(&podv2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	update = got.(kubelet.PodUpdate)
	expected = CreatePodUpdate(kubelet.SET, kubelet.ApiserverSource, expectedBoundPodv2)
	if !api.Semantic.DeepEqual(expected, update) {
		t.Fatalf("Expected %#v, Got %#v", expected, update)
	}

	fakeWatch.Delete(&podv2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	update = got.(kubelet.PodUpdate)
	expected = CreatePodUpdate(kubelet.SET, kubelet.ApiserverSource)
	if !api.Semantic.DeepEqual(expected, update) {
		t.Errorf("Expected %#v, Got %#v", expected, update)
	}
}

func TestNewSourceApiserverInitialEmptySendsEmptyPodUpdate(t *testing.T) {
	// Setup fake api client.
	fakeWatch := watch.NewFake()
	lw := fakePodLW{
		listResp:  &api.PodList{Items: []api.Pod{}},
		watchResp: fakeWatch,
	}

	ch := make(chan interface{})

	newSourceApiserverFromLW(lw, ch)

	got, ok := <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	update := got.(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET, kubelet.ApiserverSource)
	if !api.Semantic.DeepEqual(expected, update) {
		t.Errorf("Expected %#v; Got %#v", expected, update)
	}
}
