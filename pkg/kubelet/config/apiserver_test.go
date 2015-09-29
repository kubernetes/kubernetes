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

package config

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

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

func TestNewSourceApiserver_UpdatesAndMultiplePods(t *testing.T) {
	pod1v1 := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "p"},
		Spec:       api.PodSpec{Containers: []api.Container{{Image: "image/one"}}}}
	pod1v2 := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "p"},
		Spec:       api.PodSpec{Containers: []api.Container{{Image: "image/two"}}}}
	pod2 := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "q"},
		Spec:       api.PodSpec{Containers: []api.Container{{Image: "image/blah"}}}}

	// Setup fake api client.
	fakeWatch := watch.NewFake()
	lw := fakePodLW{
		listResp:  &api.PodList{Items: []api.Pod{*pod1v1}},
		watchResp: fakeWatch,
	}

	ch := make(chan interface{})

	newSourceApiserverFromLW(lw, ch)

	got, ok := <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	update := got.(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET, kubelet.ApiserverSource, pod1v1)
	if !api.Semantic.DeepEqual(expected, update) {
		t.Errorf("Expected %#v; Got %#v", expected, update)
	}

	// Add another pod
	fakeWatch.Add(pod2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	update = got.(kubelet.PodUpdate)
	// Could be sorted either of these two ways:
	expectedA := CreatePodUpdate(kubelet.SET, kubelet.ApiserverSource, pod1v1, pod2)
	expectedB := CreatePodUpdate(kubelet.SET, kubelet.ApiserverSource, pod2, pod1v1)

	if !api.Semantic.DeepEqual(expectedA, update) && !api.Semantic.DeepEqual(expectedB, update) {
		t.Errorf("Expected %#v or %#v, Got %#v", expectedA, expectedB, update)
	}

	// Modify pod1
	fakeWatch.Modify(pod1v2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	update = got.(kubelet.PodUpdate)
	expectedA = CreatePodUpdate(kubelet.SET, kubelet.ApiserverSource, pod1v2, pod2)
	expectedB = CreatePodUpdate(kubelet.SET, kubelet.ApiserverSource, pod2, pod1v2)

	if !api.Semantic.DeepEqual(expectedA, update) && !api.Semantic.DeepEqual(expectedB, update) {
		t.Errorf("Expected %#v or %#v, Got %#v", expectedA, expectedB, update)
	}

	// Delete pod1
	fakeWatch.Delete(pod1v2)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	update = got.(kubelet.PodUpdate)
	expected = CreatePodUpdate(kubelet.SET, kubelet.ApiserverSource, pod2)
	if !api.Semantic.DeepEqual(expected, update) {
		t.Errorf("Expected %#v, Got %#v", expected, update)
	}

	// Delete pod2
	fakeWatch.Delete(pod2)
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

func TestNewSourceApiserver_TwoNamespacesSameName(t *testing.T) {
	pod1 := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "p", Namespace: "one"},
		Spec:       api.PodSpec{Containers: []api.Container{{Image: "image/one"}}}}
	pod2 := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "p", Namespace: "two"},
		Spec:       api.PodSpec{Containers: []api.Container{{Image: "image/blah"}}}}

	// Setup fake api client.
	fakeWatch := watch.NewFake()
	lw := fakePodLW{
		listResp:  &api.PodList{Items: []api.Pod{pod1, pod2}},
		watchResp: fakeWatch,
	}

	ch := make(chan interface{})

	newSourceApiserverFromLW(lw, ch)

	got, ok := <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	update := got.(kubelet.PodUpdate)
	// Make sure that we get both pods.  Catches bug #2294.
	if !(len(update.Pods) == 2) {
		t.Errorf("Expected %d, Got %d", 2, len(update.Pods))
	}

	// Delete pod1
	fakeWatch.Delete(&pod1)
	got, ok = <-ch
	if !ok {
		t.Errorf("Unable to read from channel when expected")
	}
	update = got.(kubelet.PodUpdate)
	if !(len(update.Pods) == 1) {
		t.Errorf("Expected %d, Got %d", 1, len(update.Pods))
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
