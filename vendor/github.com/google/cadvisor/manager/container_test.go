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
	"sync"
	"testing"
	"time"

	"github.com/google/cadvisor/cache/memory"
	"github.com/google/cadvisor/collector"
	"github.com/google/cadvisor/container"
	containertest "github.com/google/cadvisor/container/testing"
	info "github.com/google/cadvisor/info/v1"
	itest "github.com/google/cadvisor/info/v1/test"

	"github.com/google/cadvisor/accelerators"
	"github.com/mindprince/gonvml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	clock "k8s.io/utils/clock/testing"
)

const (
	containerName        = "/container"
	testLongHousekeeping = time.Second
)

// Create a containerData instance for a test.
func setupContainerData(t *testing.T, spec info.ContainerSpec) (*containerData, *containertest.MockContainerHandler, *memory.InMemoryCache, *clock.FakeClock) {
	mockHandler := containertest.NewMockContainerHandler(containerName)
	mockHandler.On("GetSpec").Return(
		spec,
		nil,
	)
	memoryCache := memory.New(60, nil)
	fakeClock := clock.NewFakeClock(time.Now())
	ret, err := newContainerData(containerName, memoryCache, mockHandler, false, &collector.GenericCollectorManager{}, 60*time.Second, true, fakeClock)
	if err != nil {
		t.Fatal(err)
	}
	return ret, mockHandler, memoryCache, fakeClock
}

// Create a containerData instance for a test and add a default GetSpec mock.
func newTestContainerData(t *testing.T) (*containerData, *containertest.MockContainerHandler, *memory.InMemoryCache, *clock.FakeClock) {
	return setupContainerData(t, itest.GenerateRandomContainerSpec(4))
}

func TestUpdateSubcontainers(t *testing.T) {
	subcontainers := []info.ContainerReference{
		{Name: "/container/ee0103"},
		{Name: "/container/abcd"},
		{Name: "/container/something"},
	}
	cd, mockHandler, _, _ := newTestContainerData(t)
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
	cd, mockHandler, _, _ := newTestContainerData(t)
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
	cd, mockHandler, _, _ := newTestContainerData(t)
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

	cd, mockHandler, memoryCache, _ := newTestContainerData(t)
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
	cd, mockHandler, _, _ := newTestContainerData(t)
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
	cd, mockHandler, _, _ := setupContainerData(t, spec)
	mockHandler.On("ListContainers", container.ListSelf).Return(
		subcontainers,
		nil,
	)
	mockHandler.Aliases = []string{"a1", "a2"}

	info, err := cd.GetInfo(true)
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

func TestUpdateNvidiaStats(t *testing.T) {
	cd, _, _, _ := newTestContainerData(t)
	stats := info.ContainerStats{}

	// When there are no devices, we should not get an error and stats should not change.
	cd.nvidiaCollector = &accelerators.NvidiaCollector{}
	err := cd.nvidiaCollector.UpdateStats(&stats)
	assert.Nil(t, err)
	assert.Equal(t, info.ContainerStats{}, stats)

	// This is an impossible situation (there are devices but nvml is not initialized).
	// Here I am testing that the CGo gonvml library doesn't panic when passed bad
	// input and instead returns an error.
	cd.nvidiaCollector = &accelerators.NvidiaCollector{Devices: []gonvml.Device{{}, {}}}
	err = cd.nvidiaCollector.UpdateStats(&stats)
	assert.NotNil(t, err)
	assert.Equal(t, info.ContainerStats{}, stats)
}

func TestOnDemandHousekeeping(t *testing.T) {
	statsList := itest.GenerateRandomStats(1, 4, 1*time.Second)
	stats := statsList[0]

	cd, mockHandler, memoryCache, fakeClock := newTestContainerData(t)
	mockHandler.On("GetStats").Return(stats, nil)
	defer cd.Stop()

	// 0 seconds should always trigger an update
	go cd.OnDemandHousekeeping(0 * time.Second)
	cd.housekeepingTick(fakeClock.NewTimer(time.Minute).C(), testLongHousekeeping)

	fakeClock.Step(2 * time.Second)

	// This should return without requiring a housekeepingTick because stats have been updated recently enough
	cd.OnDemandHousekeeping(3 * time.Second)

	go cd.OnDemandHousekeeping(1 * time.Second)
	cd.housekeepingTick(fakeClock.NewTimer(time.Minute).C(), testLongHousekeeping)

	checkNumStats(t, memoryCache, 2)
	mockHandler.AssertExpectations(t)
}

func TestConcurrentOnDemandHousekeeping(t *testing.T) {
	statsList := itest.GenerateRandomStats(1, 4, 1*time.Second)
	stats := statsList[0]

	cd, mockHandler, memoryCache, fakeClock := newTestContainerData(t)
	mockHandler.On("GetStats").Return(stats, nil)
	defer cd.Stop()

	numConcurrentCalls := 5
	var waitForHousekeeping sync.WaitGroup
	waitForHousekeeping.Add(numConcurrentCalls)
	onDemandCache := []chan struct{}{}
	for i := 0; i < numConcurrentCalls; i++ {
		go func() {
			cd.OnDemandHousekeeping(0 * time.Second)
			waitForHousekeeping.Done()
		}()
		// Wait for work to be queued
		onDemandCache = append(onDemandCache, <-cd.onDemandChan)
	}
	// Requeue work:
	for _, ch := range onDemandCache {
		cd.onDemandChan <- ch
	}

	go cd.housekeepingTick(fakeClock.NewTimer(time.Minute).C(), testLongHousekeeping)
	// Ensure that all queued calls return with only a single call to housekeepingTick
	waitForHousekeeping.Wait()

	checkNumStats(t, memoryCache, 1)
	mockHandler.AssertExpectations(t)
}
