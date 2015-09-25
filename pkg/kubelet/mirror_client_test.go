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

package kubelet

import (
	"sync"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/sets"
)

type fakeMirrorClient struct {
	mirrorPodLock sync.RWMutex
	// Note that a real mirror manager does not store the mirror pods in
	// itself. This fake manager does this to track calls.
	mirrorPods   sets.String
	createCounts map[string]int
	deleteCounts map[string]int
}

func (fmc *fakeMirrorClient) CreateMirrorPod(pod *api.Pod) error {
	fmc.mirrorPodLock.Lock()
	defer fmc.mirrorPodLock.Unlock()
	podFullName := kubecontainer.GetPodFullName(pod)
	fmc.mirrorPods.Insert(podFullName)
	fmc.createCounts[podFullName]++
	return nil
}

func (fmc *fakeMirrorClient) DeleteMirrorPod(podFullName string) error {
	fmc.mirrorPodLock.Lock()
	defer fmc.mirrorPodLock.Unlock()
	fmc.mirrorPods.Delete(podFullName)
	fmc.deleteCounts[podFullName]++
	return nil
}

func newFakeMirrorClient() *fakeMirrorClient {
	m := fakeMirrorClient{}
	m.mirrorPods = sets.NewString()
	m.createCounts = make(map[string]int)
	m.deleteCounts = make(map[string]int)
	return &m
}

func (fmc *fakeMirrorClient) HasPod(podFullName string) bool {
	fmc.mirrorPodLock.RLock()
	defer fmc.mirrorPodLock.RUnlock()
	return fmc.mirrorPods.Has(podFullName)
}

func (fmc *fakeMirrorClient) NumOfPods() int {
	fmc.mirrorPodLock.RLock()
	defer fmc.mirrorPodLock.RUnlock()
	return fmc.mirrorPods.Len()
}

func (fmc *fakeMirrorClient) GetPods() []string {
	fmc.mirrorPodLock.RLock()
	defer fmc.mirrorPodLock.RUnlock()
	return fmc.mirrorPods.List()
}

func (fmc *fakeMirrorClient) GetCounts(podFullName string) (int, int) {
	fmc.mirrorPodLock.RLock()
	defer fmc.mirrorPodLock.RUnlock()
	return fmc.createCounts[podFullName], fmc.deleteCounts[podFullName]
}

func TestParsePodFullName(t *testing.T) {
	type nameTuple struct {
		Name      string
		Namespace string
	}
	successfulCases := map[string]nameTuple{
		"bar_foo":         {Name: "bar", Namespace: "foo"},
		"bar.org_foo.com": {Name: "bar.org", Namespace: "foo.com"},
		"bar-bar_foo":     {Name: "bar-bar", Namespace: "foo"},
	}
	failedCases := []string{"barfoo", "bar_foo_foo", ""}

	for podFullName, expected := range successfulCases {
		name, namespace, err := kubecontainer.ParsePodFullName(podFullName)
		if err != nil {
			t.Errorf("unexpected error when parsing the full name: %v", err)
			continue
		}
		if name != expected.Name || namespace != expected.Namespace {
			t.Errorf("expected name %q, namespace %q; got name %q, namespace %q",
				expected.Name, expected.Namespace, name, namespace)
		}
	}
	for _, podFullName := range failedCases {
		_, _, err := kubecontainer.ParsePodFullName(podFullName)
		if err == nil {
			t.Errorf("expected error when parsing the full name, got none")
		}
	}
}
