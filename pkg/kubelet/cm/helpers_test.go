/*
Copyright 2024 The Kubernetes Authors.

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

package cm

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	apitest "k8s.io/cri-api/pkg/apis/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
)

func makeFakePodSandbox(name, namespace string) *apitest.FakePodSandbox {
	uid := fmt.Sprintf("pod-innerId-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", ""))
	return &apitest.FakePodSandbox{
		PodSandboxStatus: runtimeapi.PodSandboxStatus{
			Metadata: &runtimeapi.PodSandboxMetadata{
				Name:      name,
				Uid:       uid,
				Namespace: namespace,
			},
			Id:        uid,
			State:     runtimeapi.PodSandboxState_SANDBOX_READY,
			CreatedAt: time.Now().UnixNano(),
		},
	}
}

func makeFakeContainer(sandbox *apitest.FakePodSandbox, name string, attempt uint32, terminated bool, createTime int, containerID string) *apitest.FakeContainer {
	sandboxID := sandbox.PodSandboxStatus.Id
	c := &apitest.FakeContainer{
		SandboxID: sandboxID,
		ContainerStatus: runtimeapi.ContainerStatus{
			Id:        containerID,
			Metadata:  &runtimeapi.ContainerMetadata{Name: name, Attempt: attempt},
			Image:     &runtimeapi.ImageSpec{},
			ImageRef:  "fake-image-ref",
			CreatedAt: int64(createTime),
			Labels: map[string]string{
				"io.kubernetes.pod.name":       sandbox.Metadata.Name,
				"io.kubernetes.pod.uid":        sandbox.Metadata.Uid,
				"io.kubernetes.pod.namespace":  sandbox.Metadata.Namespace,
				"io.kubernetes.container.name": name,
			},
			State: runtimeapi.ContainerState_CONTAINER_RUNNING,
		},
	}
	if terminated {
		c.State = runtimeapi.ContainerState_CONTAINER_EXITED
	}
	return c
}
func TestBuildContainerMapAndRunningSetFromRuntime(t *testing.T) {
	pod1 := makeFakePodSandbox("pod1", "default")
	pod1container1 := makeFakeContainer(pod1, "test1", 0, true, time.Now().Add(-time.Second*50).Nanosecond(), fmt.Sprintf("containerId0-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", "")))
	pod1container2 := makeFakeContainer(pod1, "test1", 1, true, time.Now().Add(-time.Second*40).Nanosecond(), fmt.Sprintf("containerId0-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", "")))
	pod1container3 := makeFakeContainer(pod1, "test1", 2, true, time.Now().Add(-time.Second*30).Nanosecond(), fmt.Sprintf("containerId0-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", "")))
	pod1container4 := makeFakeContainer(pod1, "test1", 3, true, time.Now().Add(-time.Second*20).Nanosecond(), fmt.Sprintf("containerId0-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", "")))
	pod1container5 := makeFakeContainer(pod1, "test1", 4, false, time.Now().Nanosecond(), fmt.Sprintf("containerId1-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", "")))

	pod2 := makeFakePodSandbox("pod2", "default")
	pod2container1 := makeFakeContainer(pod2, "test1", 0, true, time.Now().Add(-time.Second*50).Nanosecond(), fmt.Sprintf("containerId0-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", "")))
	pod2container2 := makeFakeContainer(pod2, "test1", 1, true, time.Now().Add(-time.Second*40).Nanosecond(), fmt.Sprintf("containerId0-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", "")))
	pod2container3 := makeFakeContainer(pod2, "test1", 2, true, time.Now().Add(-time.Second*30).Nanosecond(), fmt.Sprintf("containerId0-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", "")))
	pod2container4 := makeFakeContainer(pod2, "test1", 3, true, time.Now().Add(-time.Second*20).Nanosecond(), fmt.Sprintf("containerId0-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", "")))
	pod2container5 := makeFakeContainer(pod2, "test1", 4, true, time.Now().Add(-time.Second*10).Nanosecond(), fmt.Sprintf("containerId0-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", "")))
	pod2container6 := makeFakeContainer(pod2, "test1", 5, false, time.Now().Nanosecond(), fmt.Sprintf("containerId1-%s", strings.ReplaceAll(string(uuid.NewUUID()), "-", "")))

	testcases := []struct {
		name               string
		pods               []*apitest.FakePodSandbox
		containers         []*apitest.FakeContainer
		expectContainerMap containermap.ContainerMap
		expectRunningSet   sets.Set[string]
	}{
		{
			name: "one pod,one container",
			pods: []*apitest.FakePodSandbox{
				pod1,
			},
			containers: []*apitest.FakeContainer{
				pod1container5,
			},
			expectContainerMap: func() containermap.ContainerMap {
				cm := containermap.NewContainerMap()
				cm.Add(pod1.Id, pod1container5.Metadata.Name, pod1container5.Id)
				return cm
			}(),
			expectRunningSet: func() sets.Set[string] {
				return sets.New(pod1container5.Id)
			}(),
		},
		{
			name: "one pod,multi container",
			pods: []*apitest.FakePodSandbox{
				pod1,
			},
			containers: []*apitest.FakeContainer{
				// descending order, to check if the function will sort
				pod1container5, pod1container4, pod1container3, pod1container2, pod1container1,
			},
			expectContainerMap: func() containermap.ContainerMap {
				cm := containermap.NewContainerMap()
				cm.Add(pod1.Id, pod1container5.Metadata.Name, pod1container5.Id)
				return cm
			}(),
			expectRunningSet: func() sets.Set[string] {
				return sets.New(pod1container5.Id)
			}(),
		},
		{
			name: "multi pod,one container",
			pods: []*apitest.FakePodSandbox{
				pod1, pod2,
			},
			containers: []*apitest.FakeContainer{
				pod1container5, pod2container6,
			},
			expectContainerMap: func() containermap.ContainerMap {
				cm := containermap.NewContainerMap()
				cm.Add(pod1.Id, pod1container5.Metadata.Name, pod1container5.Id)
				cm.Add(pod2.Id, pod2container6.Metadata.Name, pod2container6.Id)
				return cm
			}(),
			expectRunningSet: func() sets.Set[string] {
				return sets.New(pod1container5.Id, pod2container6.Id)
			}(),
		},
		{
			name: "multi pod,multi container",
			pods: []*apitest.FakePodSandbox{
				pod1, pod2,
			},
			containers: []*apitest.FakeContainer{
				pod1container5, pod1container4, pod1container3, pod1container2, pod1container1,
				pod2container6, pod2container5, pod2container4, pod2container3, pod2container2, pod2container1,
			},
			expectContainerMap: func() containermap.ContainerMap {
				cm := containermap.NewContainerMap()
				cm.Add(pod1.Id, pod1container5.Metadata.Name, pod1container5.Id)
				cm.Add(pod2.Id, pod2container6.Metadata.Name, pod2container6.Id)
				return cm
			}(),
			expectRunningSet: func() sets.Set[string] {
				return sets.New(pod1container5.Id, pod2container6.Id)
			}(),
		},
	}

	for _, ts := range testcases {
		fakeRuntimeService := apitest.NewFakeRuntimeService()

		sandboxes := map[string]*apitest.FakePodSandbox{}
		for _, pod := range ts.pods {
			sandboxes[pod.Id] = pod
		}
		fakeRuntimeService.Sandboxes = sandboxes

		containers := map[string]*apitest.FakeContainer{}
		for _, container := range ts.containers {
			containers[container.Id] = container
		}
		fakeRuntimeService.Containers = containers

		gotContainerMap, gotRunningSet := buildContainerMapAndRunningSetFromRuntime(ctx, fakeRuntimeService)
		if !reflect.DeepEqual(gotContainerMap, ts.expectContainerMap) {
			t.Errorf("%s:ContainerMap is wrong, got:%v,expect:%v", ts.name, gotContainerMap, ts.expectContainerMap)
		}
		if !reflect.DeepEqual(gotRunningSet, ts.expectRunningSet) {
			t.Errorf("%s:RunningSet is wrong, got:%v,expect:%v", ts.name, gotRunningSet, ts.expectRunningSet)
		}
	}

}
