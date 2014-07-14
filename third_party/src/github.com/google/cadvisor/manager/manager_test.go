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
	"reflect"
	"testing"
	"time"

	"github.com/google/cadvisor/container"
	ctest "github.com/google/cadvisor/container/test"
	"github.com/google/cadvisor/info"
	itest "github.com/google/cadvisor/info/test"
	stest "github.com/google/cadvisor/storage/test"
)

func createManagerAndAddContainers(
	driver *stest.MockStorageDriver,
	containers []string,
	f func(*ctest.MockContainerHandler),
	t *testing.T,
) *manager {
	if driver == nil {
		driver = &stest.MockStorageDriver{}
	}
	factory := &ctest.FactoryForMockContainerHandler{
		Name: "factoryForManager",
		PrepareContainerHandlerFunc: func(name string, handler *ctest.MockContainerHandler) {
			handler.Name = name
			found := false
			for _, c := range containers {
				if c == name {
					found = true
				}
			}
			if !found {
				t.Errorf("Asked to create a container with name %v, which is unknown.", name)
			}
			f(handler)
		},
	}
	container.RegisterContainerHandlerFactory("/", factory)
	mif, err := New(driver)
	if err != nil {
		t.Fatal(err)
	}
	if ret, ok := mif.(*manager); ok {
		for _, container := range containers {
			ret.containers[container], err = NewContainerData(container, driver)
			if err != nil {
				t.Fatal(err)
			}
		}
		return ret
	}
	t.Fatal("Wrong type")
	return nil
}

func TestGetContainerInfo(t *testing.T) {
	containers := []string{
		"/c1",
		"/c2",
	}

	query := &info.ContainerInfoRequest{
		NumStats:               256,
		NumSamples:             128,
		CpuUsagePercentiles:    []int{10, 50, 90},
		MemoryUsagePercentages: []int{10, 80, 90},
	}

	infosMap := make(map[string]*info.ContainerInfo, len(containers))
	handlerMap := make(map[string]*ctest.MockContainerHandler, len(containers))

	for _, container := range containers {
		infosMap[container] = itest.GenerateRandomContainerInfo(container, 4, query, 1*time.Second)
	}

	driver := &stest.MockStorageDriver{}
	m := createManagerAndAddContainers(
		driver,
		containers,
		func(h *ctest.MockContainerHandler) {
			cinfo := infosMap[h.Name]
			stats := cinfo.Stats
			samples := cinfo.Samples
			percentiles := cinfo.StatsPercentiles
			spec := cinfo.Spec
			driver.On(
				"Percentiles",
				h.Name,
				query.CpuUsagePercentiles,
				query.MemoryUsagePercentages,
			).Return(
				percentiles,
				nil,
			)

			driver.On(
				"Samples",
				h.Name,
				query.NumSamples,
			).Return(
				samples,
				nil,
			)

			driver.On(
				"RecentStats",
				h.Name,
				query.NumStats,
			).Return(
				stats,
				nil,
			)

			h.On("ListContainers", container.LIST_SELF).Return(
				[]info.ContainerReference(nil),
				nil,
			)
			h.On("GetSpec").Return(
				spec,
				nil,
			)
			handlerMap[h.Name] = h
		},
		t,
	)

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

func TestGetContainerInfoWithDefaultValue(t *testing.T) {
	containers := []string{
		"/c1",
		"/c2",
	}

	var query *info.ContainerInfoRequest
	query = query.FillDefaults()

	infosMap := make(map[string]*info.ContainerInfo, len(containers))
	handlerMap := make(map[string]*ctest.MockContainerHandler, len(containers))

	for _, container := range containers {
		infosMap[container] = itest.GenerateRandomContainerInfo(container, 4, query, 1*time.Second)
	}

	driver := &stest.MockStorageDriver{}
	m := createManagerAndAddContainers(
		driver,
		containers,
		func(h *ctest.MockContainerHandler) {
			cinfo := infosMap[h.Name]
			stats := cinfo.Stats
			samples := cinfo.Samples
			percentiles := cinfo.StatsPercentiles
			spec := cinfo.Spec
			driver.On(
				"Percentiles",
				h.Name,
				query.CpuUsagePercentiles,
				query.MemoryUsagePercentages,
			).Return(
				percentiles,
				nil,
			)

			driver.On(
				"Samples",
				h.Name,
				query.NumSamples,
			).Return(
				samples,
				nil,
			)

			driver.On(
				"RecentStats",
				h.Name,
				query.NumStats,
			).Return(
				stats,
				nil,
			)

			h.On("ListContainers", container.LIST_SELF).Return(
				[]info.ContainerReference(nil),
				nil,
			)
			h.On("GetSpec").Return(
				spec,
				nil,
			)
			handlerMap[h.Name] = h
		},
		t,
	)

	returnedInfos := make(map[string]*info.ContainerInfo, len(containers))

	for _, container := range containers {
		// nil should give us default values
		cinfo, err := m.GetContainerInfo(container, nil)
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
