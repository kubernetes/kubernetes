/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	"sync"

	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/sets"
)

type FakeMirrorClient struct {
	mirrorPodLock sync.RWMutex
	// Note that a real mirror manager does not store the mirror pods in
	// itself. This fake manager does this to track calls.
	mirrorPods   sets.String
	createCounts map[string]int
	deleteCounts map[string]int
}

func NewFakeMirrorClient() *FakeMirrorClient {
	m := FakeMirrorClient{}
	m.mirrorPods = sets.NewString()
	m.createCounts = make(map[string]int)
	m.deleteCounts = make(map[string]int)
	return &m
}

func (fmc *FakeMirrorClient) CreateMirrorPod(pod *api.Pod) error {
	fmc.mirrorPodLock.Lock()
	defer fmc.mirrorPodLock.Unlock()
	podFullName := kubecontainer.GetPodFullName(pod)
	fmc.mirrorPods.Insert(podFullName)
	fmc.createCounts[podFullName]++
	return nil
}

func (fmc *FakeMirrorClient) DeleteMirrorPod(podFullName string) error {
	fmc.mirrorPodLock.Lock()
	defer fmc.mirrorPodLock.Unlock()
	fmc.mirrorPods.Delete(podFullName)
	fmc.deleteCounts[podFullName]++
	return nil
}

func (fmc *FakeMirrorClient) HasPod(podFullName string) bool {
	fmc.mirrorPodLock.RLock()
	defer fmc.mirrorPodLock.RUnlock()
	return fmc.mirrorPods.Has(podFullName)
}

func (fmc *FakeMirrorClient) NumOfPods() int {
	fmc.mirrorPodLock.RLock()
	defer fmc.mirrorPodLock.RUnlock()
	return fmc.mirrorPods.Len()
}

func (fmc *FakeMirrorClient) GetPods() []string {
	fmc.mirrorPodLock.RLock()
	defer fmc.mirrorPodLock.RUnlock()
	return fmc.mirrorPods.List()
}

func (fmc *FakeMirrorClient) GetCounts(podFullName string) (int, int) {
	fmc.mirrorPodLock.RLock()
	defer fmc.mirrorPodLock.RUnlock()
	return fmc.createCounts[podFullName], fmc.deleteCounts[podFullName]
}
