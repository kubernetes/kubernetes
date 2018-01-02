// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Per-container manager.

package manager

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"net/http"

	"github.com/google/cadvisor/cache/memory"
	"github.com/google/cadvisor/collector"
	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/docker"
	containertest "github.com/google/cadvisor/container/testing"
	info "github.com/google/cadvisor/info/v1"
	itest "github.com/google/cadvisor/info/v1/test"
	"github.com/google/cadvisor/info/v2"
	"github.com/google/cadvisor/utils/sysfs/fakesysfs"

	"github.com/stretchr/testify/assert"
	clock "k8s.io/utils/clock/testing"
)

// TODO(vmarmol): Refactor these tests.

func createManagerAndAddContainers(
	memoryCache *memory.InMemoryCache,
	sysfs *fakesysfs.FakeSysFs,
	containers []string,
	f func(*containertest.MockContainerHandler),
	t *testing.T,
) *manager {
	container.ClearContainerHandlerFactories()
	mif := &manager{
		containers:   make(map[namespacedContainerName]*containerData),
		quitChannels: make([]chan error, 0, 2),
		memoryCache:  memoryCache,
	}
	for _, name := range containers {
		mockHandler := containertest.NewMockContainerHandler(name)
		spec := itest.GenerateRandomContainerSpec(4)
		mockHandler.On("GetSpec").Return(
			spec,
			nil,
		).Once()
		cont, err := newContainerData(name, memoryCache, mockHandler, false, &collector.GenericCollectorManager{}, 60*time.Second, true, clock.NewFakeClock(time.Now()))
		if err != nil {
			t.Fatal(err)
		}
		mif.containers[namespacedContainerName{
			Name: name,
		}] = cont
		// Add Docker containers under their namespace.
		if strings.HasPrefix(name, "/docker") {
			mif.containers[namespacedContainerName{
				Namespace: docker.DockerNamespace,
				Name:      strings.TrimPrefix(name, "/docker/"),
			}] = cont
		}
		f(mockHandler)
	}
	return mif
}

// Expect a manager with the specified containers and query. Returns the manager, map of ContainerInfo objects,
// and map of MockContainerHandler objects.}
func expectManagerWithContainers(containers []string, query *info.ContainerInfoRequest, t *testing.T) (*manager, map[string]*info.ContainerInfo, map[string]*containertest.MockContainerHandler) {
	infosMap := make(map[string]*info.ContainerInfo, len(containers))
	handlerMap := make(map[string]*containertest.MockContainerHandler, len(containers))

	for _, container := range containers {
		infosMap[container] = itest.GenerateRandomContainerInfo(container, 4, query, 1*time.Second)
	}

	memoryCache := memory.New(time.Duration(query.NumStats)*time.Second, nil)
	sysfs := &fakesysfs.FakeSysFs{}
	m := createManagerAndAddContainers(
		memoryCache,
		sysfs,
		containers,
		func(h *containertest.MockContainerHandler) {
			cinfo := infosMap[h.Name]
			ref, err := h.ContainerReference()
			if err != nil {
				t.Error(err)
			}
			for _, stat := range cinfo.Stats {
				err = memoryCache.AddStats(ref, stat)
				if err != nil {
					t.Error(err)
				}
			}
			spec := cinfo.Spec

			h.On("ListContainers", container.ListSelf).Return(
				[]info.ContainerReference(nil),
				nil,
			)
			h.On("GetSpec").Return(
				spec,
				nil,
			).Once()
			handlerMap[h.Name] = h
		},
		t,
	)

	return m, infosMap, handlerMap
}

// Expect a manager with the specified containers and query. Returns the manager, map of ContainerInfo objects,
// and map of MockContainerHandler objects.}
func expectManagerWithContainersV2(containers []string, query *info.ContainerInfoRequest, t *testing.T) (*manager, map[string]*info.ContainerInfo, map[string]*containertest.MockContainerHandler) {
	infosMap := make(map[string]*info.ContainerInfo, len(containers))
	handlerMap := make(map[string]*containertest.MockContainerHandler, len(containers))

	for _, container := range containers {
		infosMap[container] = itest.GenerateRandomContainerInfo(container, 4, query, 1*time.Second)
	}

	memoryCache := memory.New(time.Duration(query.NumStats)*time.Second, nil)
	sysfs := &fakesysfs.FakeSysFs{}
	m := createManagerAndAddContainers(
		memoryCache,
		sysfs,
		containers,
		func(h *containertest.MockContainerHandler) {
			cinfo := infosMap[h.Name]
			ref, err := h.ContainerReference()
			if err != nil {
				t.Error(err)
			}
			for _, stat := range cinfo.Stats {
				err = memoryCache.AddStats(ref, stat)
				if err != nil {
					t.Error(err)
				}
			}
			spec := cinfo.Spec

			h.On("GetSpec").Return(
				spec,
				nil,
			).Once()
			handlerMap[h.Name] = h
		},
		t,
	)

	return m, infosMap, handlerMap
}

func TestGetContainerInfo(t *testing.T) {
	containers := []string{
		"/c1",
		"/c2",
	}

	query := &info.ContainerInfoRequest{
		NumStats: 256,
	}

	m, infosMap, handlerMap := expectManagerWithContainers(containers, query, t)

	returnedInfos := make(map[string]*info.ContainerInfo, len(containers))

	for _, container := range containers {
		cinfo, err := m.GetContainerInfo(container, query)
		if err != nil {
			t.Fatalf("Unable to get info for container %v: %v", container, err)
		}
		returnedInfos[container] = cinfo
	}

	for container, handler := range handlerMap {
		handler.AssertExpectations(t)
		returned := returnedInfos[container]
		expected := infosMap[container]
		if !reflect.DeepEqual(returned, expected) {
			t.Errorf("returned unexpected info for container %v; returned %+v; expected %+v", container, returned, expected)
		}
	}

}

func TestGetContainerInfoV2(t *testing.T) {
	containers := []string{
		"/",
		"/c1",
		"/c2",
	}

	options := v2.RequestOptions{
		IdType:    v2.TypeName,
		Count:     1,
		Recursive: true,
	}
	query := &info.ContainerInfoRequest{
		NumStats: 2,
	}

	m, _, handlerMap := expectManagerWithContainersV2(containers, query, t)

	infos, err := m.GetContainerInfoV2("/", options)
	if err != nil {
		t.Fatalf("GetContainerInfoV2 failed: %v", err)
	}

	for container, handler := range handlerMap {
		handler.AssertExpectations(t)
		info, ok := infos[container]
		assert.True(t, ok, "Missing info for container %q", container)
		assert.NotEqual(t, v2.ContainerSpec{}, info.Spec, "Empty spec for container %q", container)
		assert.NotEmpty(t, info.Stats, "Missing stats for container %q", container)
	}
}

func TestGetContainerInfoV2Failure(t *testing.T) {
	successful := "/"
	statless := "/c1"
	failing := "/c2"
	containers := []string{
		successful, statless, failing,
	}

	options := v2.RequestOptions{
		IdType:    v2.TypeName,
		Count:     1,
		Recursive: true,
	}
	query := &info.ContainerInfoRequest{
		NumStats: 2,
	}

	m, _, handlerMap := expectManagerWithContainers(containers, query, t)

	// Remove /c1 stats
	err := m.memoryCache.RemoveContainer(statless)
	if err != nil {
		t.Fatalf("RemoveContainer failed: %v", err)
	}

	// Make GetSpec fail on /c2
	mockErr := fmt.Errorf("intentional GetSpec failure")
	handlerMap[failing].GetSpec() // Use up default GetSpec call, and replace below
	handlerMap[failing].On("GetSpec").Return(info.ContainerSpec{}, mockErr)
	handlerMap[failing].On("Exists").Return(true)
	m.containers[namespacedContainerName{Name: failing}].infoLastUpdatedTime = time.Time{} // Force GetSpec.

	infos, err := m.GetContainerInfoV2("/", options)
	if err == nil {
		t.Error("Expected error calling GetContainerInfoV2")
	}

	// Successful containers still successful.
	info, ok := infos[successful]
	assert.True(t, ok, "Missing info for container %q", successful)
	assert.NotEqual(t, v2.ContainerSpec{}, info.Spec, "Empty spec for container %q", successful)
	assert.NotEmpty(t, info.Stats, "Missing stats for container %q", successful)

	// "/c1" present with spec.
	info, ok = infos[statless]
	assert.True(t, ok, "Missing info for container %q", statless)
	assert.NotEqual(t, v2.ContainerSpec{}, info.Spec, "Empty spec for container %q", statless)
	assert.Empty(t, info.Stats, "Missing stats for container %q", successful)

	// "/c2" should be present but empty.
	info, ok = infos[failing]
	assert.True(t, ok, "Missing info for failed container")
	assert.Equal(t, v2.ContainerInfo{}, info, "Empty spec for failed container")
	assert.Empty(t, info.Stats, "Missing stats for failed container")
}

func TestSubcontainersInfo(t *testing.T) {
	containers := []string{
		"/c1",
		"/c2",
	}

	query := &info.ContainerInfoRequest{
		NumStats: 64,
	}

	m, _, _ := expectManagerWithContainers(containers, query, t)

	result, err := m.SubcontainersInfo("/", query)
	if err != nil {
		t.Fatalf("expected to succeed: %s", err)
	}
	if len(result) != len(containers) {
		t.Errorf("expected to received containers: %v, but received: %v", containers, result)
	}
	for _, res := range result {
		found := false
		for _, name := range containers {
			if res.Name == name {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("unexpected container %q in result, expected one of %v", res.Name, containers)
		}
	}
}

func TestDockerContainersInfo(t *testing.T) {
	containers := []string{
		"/docker/c1a",
		"/docker/c2a",
	}

	query := &info.ContainerInfoRequest{
		NumStats: 2,
	}

	m, _, _ := expectManagerWithContainers(containers, query, t)

	result, err := m.DockerContainer("c1a", query)
	if err != nil {
		t.Fatalf("expected to succeed: %s", err)
	}
	if result.Name != containers[0] {
		t.Errorf("Unexpected container %q in result. Expected container %q", result.Name, containers[0])
	}

	result, err = m.DockerContainer("c2", query)
	if err != nil {
		t.Fatalf("expected to succeed: %s", err)
	}
	if result.Name != containers[1] {
		t.Errorf("Unexpected container %q in result. Expected container %q", result.Name, containers[1])
	}

	result, err = m.DockerContainer("c", query)
	expectedError := "unable to find container. Container \"c\" is not unique"
	if err == nil {
		t.Errorf("expected error %q but received %q", expectedError, err)
	}
}

func TestNewNilManager(t *testing.T) {
	_, err := New(nil, nil, 60*time.Second, true, container.MetricSet{}, http.DefaultClient)
	if err == nil {
		t.Fatalf("Expected nil manager to return error")
	}
}
