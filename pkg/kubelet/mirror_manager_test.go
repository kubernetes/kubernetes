/*
Copyright 2015 Google Inc. All rights reserved.

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
	"reflect"
	"sync"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type fakeMirrorManager struct {
	mirrorPodLock sync.RWMutex
	// Note that a real mirror manager does not store the mirror pods in
	// itself. This fake manager does this to track calls.
	mirrorPods   util.StringSet
	createCounts map[string]int
	deleteCounts map[string]int
}

func (self *fakeMirrorManager) CreateMirrorPod(pod api.Pod, _ string) error {
	self.mirrorPodLock.Lock()
	defer self.mirrorPodLock.Unlock()
	podFullName := GetPodFullName(&pod)
	self.mirrorPods.Insert(podFullName)
	self.createCounts[podFullName]++
	return nil
}

func (self *fakeMirrorManager) DeleteMirrorPod(podFullName string) error {
	self.mirrorPodLock.Lock()
	defer self.mirrorPodLock.Unlock()
	self.mirrorPods.Delete(podFullName)
	self.deleteCounts[podFullName]++
	return nil
}

func newFakeMirrorMananger() *fakeMirrorManager {
	m := fakeMirrorManager{}
	m.mirrorPods = util.NewStringSet()
	m.createCounts = make(map[string]int)
	m.deleteCounts = make(map[string]int)
	return &m
}

func (self *fakeMirrorManager) HasPod(podFullName string) bool {
	self.mirrorPodLock.RLock()
	defer self.mirrorPodLock.RUnlock()
	return self.mirrorPods.Has(podFullName)
}

func (self *fakeMirrorManager) NumOfPods() int {
	self.mirrorPodLock.RLock()
	defer self.mirrorPodLock.RUnlock()
	return self.mirrorPods.Len()
}

func (self *fakeMirrorManager) GetPods() []string {
	self.mirrorPodLock.RLock()
	defer self.mirrorPodLock.RUnlock()
	return self.mirrorPods.List()
}

func (self *fakeMirrorManager) GetCounts(podFullName string) (int, int) {
	self.mirrorPodLock.RLock()
	defer self.mirrorPodLock.RUnlock()
	return self.createCounts[podFullName], self.deleteCounts[podFullName]
}

// Tests that mirror pods are filtered out properly from the pod update.
func TestFilterOutMirrorPods(t *testing.T) {
	mirrorPod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "987654321",
			Name:      "bar",
			Namespace: "default",
			Annotations: map[string]string{
				ConfigSourceAnnotationKey: "api",
				ConfigMirrorAnnotationKey: "mirror",
			},
		},
	}
	staticPod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:         "123456789",
			Name:        "bar",
			Namespace:   "default",
			Annotations: map[string]string{ConfigSourceAnnotationKey: "file"},
		},
	}

	expectedPods := []api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:         "999999999",
				Name:        "taco",
				Namespace:   "default",
				Annotations: map[string]string{ConfigSourceAnnotationKey: "api"},
			},
		},
		staticPod,
	}
	updates := append(expectedPods, mirrorPod)
	actualPods, actualMirrorPods := filterAndCategorizePods(updates)
	if !reflect.DeepEqual(expectedPods, actualPods) {
		t.Errorf("expected %#v, got %#v", expectedPods, actualPods)
	}
	if _, ok := actualMirrorPods.mirror[GetPodFullName(&mirrorPod)]; !ok {
		t.Errorf("mirror pod is not recorded")
	}
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
		name, namespace, err := ParsePodFullName(podFullName)
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
		_, _, err := ParsePodFullName(podFullName)
		if err == nil {
			t.Errorf("expected error when parsing the full name, got none")
		}
	}
}
