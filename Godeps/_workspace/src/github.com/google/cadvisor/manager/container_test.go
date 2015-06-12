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
	"testing"
	"time"

	"github.com/google/cadvisor/cache/memory"
	"github.com/google/cadvisor/collector"
	"github.com/google/cadvisor/container"
	info "github.com/google/cadvisor/info/v1"
	itest "github.com/google/cadvisor/info/v1/test"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const containerName = "/container"

// Create a containerData instance for a test.
func setupContainerData(t *testing.T, spec info.ContainerSpec) (*containerData, *container.MockContainerHandler, *memory.InMemoryCache) {
	mockHandler := container.NewMockContainerHandler(containerName)
	mockHandler.On("GetSpec").Return(
		spec,
		nil,
	)
	memoryCache := memory.New(60, nil)
	ret, err := newContainerData(containerName, memoryCache, mockHandler, nil, false, &collector.FakeCollectorManager{})
	if err != nil {
		t.Fatal(err)
	}
	return ret, mockHandler, memoryCache
}

// Create a containerData instance for a test and add a default GetSpec mock.
func newTestContainerData(t *testing.T) (*containerData, *container.MockContainerHandler, *memory.InMemoryCache) {
	spec := itest.GenerateRandomContainerSpec(4)
	ret, mockHandler, memoryCache := setupContainerData(t, spec)
	return ret, mockHandler, memoryCache
}

func TestUpdateSubcontainers(t *testing.T) {
	subcontainers := []info.ContainerReference{
		{Name: "/container/ee0103"},
		{Name: "/container/abcd"},
		{Name: "/container/something"},
	}
	cd, mockHandler, _ := newTestContainerData(t)
	mockHandler.On("ListContainers", container.ListSelf).Return(
		subcontainers,
		nil,
	)

	err := cd.updateSubcontainers()
	if err != nil {
		t.Fatal(err)
	}

	if len(cd.info.Subcontainers) != len(subcontainers) {
		t.Errorf("Received %v subcontainers, should be %v", len(cd.info.Subcontainers), len(subcontainers))
	}

	for _, sub := range cd.info.Subcontainers {
		found := false
		for _, sub2 := range subcontainers {
			if sub.Name == sub2.Name {
				found = true
			}
		}
		if !found {
			t.Errorf("Received unknown sub container %v", sub)
		}
	}

	mockHandler.AssertExpectations(t)
}

func TestUpdateSubcontainersWithError(t *testing.T) {
	cd, mockHandler, _ := newTestContainerData(t)
	mockHandler.On("ListContainers", container.ListSelf).Return(
		[]info.ContainerReference{},
		fmt.Errorf("some error"),
	)
	mockHandler.On("Exists").Return(true)

	assert.NotNil(t, cd.updateSubcontainers())
	assert.Empty(t, cd.info.Subcontainers, "subcontainers should not be populated on failure")
	mockHandler.AssertExpectations(t)
}

func TestUpdateSubcontainersWithErrorOnDeadContainer(t *testing.T) {
	cd, mockHandler, _ := newTestContainerData(t)
	mockHandler.On("ListContainers", container.ListSelf).Return(
		[]info.ContainerReference{},
		fmt.Errorf("some error"),
	)
	mockHandler.On("Exists").Return(false)

	assert.Nil(t, cd.updateSubcontainers())
	mockHandler.AssertExpectations(t)
}

func checkNumStats(t *testing.T, memoryCache *memory.InMemoryCache, numStats int) {
	var empty time.Time
	stats, err := memoryCache.RecentStats(containerName, empty, empty, -1)
	require.Nil(t, err)
	assert.Len(t, stats, numStats)
}

func TestUpdateStats(t *testing.T) {
	statsList := itest.GenerateRandomStats(1, 4, 1*time.Second)
	stats := statsList[0]

	cd, mockHandler, memoryCache := newTestContainerData(t)
	mockHandler.On("GetStats").Return(
		stats,
		nil,
	)

	err := cd.updateStats()
	if err != nil {
		t.Fatal(err)
	}

	checkNumStats(t, memoryCache, 1)
	mockHandler.AssertExpectations(t)
}

func TestUpdateSpec(t *testing.T) {
	spec := itest.GenerateRandomContainerSpec(4)
	cd, mockHandler, _ := newTestContainerData(t)
	mockHandler.On("GetSpec").Return(
		spec,
		nil,
	)

	err := cd.updateSpec()
	if err != nil {
		t.Fatal(err)
	}

	mockHandler.AssertExpectations(t)
}

func TestGetInfo(t *testing.T) {
	spec := itest.GenerateRandomContainerSpec(4)
	subcontainers := []info.ContainerReference{
		{Name: "/container/ee0103"},
		{Name: "/container/abcd"},
		{Name: "/container/something"},
	}
	cd, mockHandler, _ := setupContainerData(t, spec)
	mockHandler.On("ListContainers", container.ListSelf).Return(
		subcontainers,
		nil,
	)
	mockHandler.Aliases = []string{"a1", "a2"}

	info, err := cd.GetInfo()
	if err != nil {
		t.Fatal(err)
	}

	mockHandler.AssertExpectations(t)

	if len(info.Subcontainers) != len(subcontainers) {
		t.Errorf("Received %v subcontainers, should be %v", len(info.Subcontainers), len(subcontainers))
	}

	for _, sub := range info.Subcontainers {
		found := false
		for _, sub2 := range subcontainers {
			if sub.Name == sub2.Name {
				found = true
			}
		}
		if !found {
			t.Errorf("Received unknown sub container %v", sub)
		}
	}

	if !reflect.DeepEqual(spec, info.Spec) {
		t.Errorf("received wrong container spec")
	}

	if info.Name != mockHandler.Name {
		t.Errorf("received wrong container name: received %v; should be %v", info.Name, mockHandler.Name)
	}
}
