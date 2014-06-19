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

package container

import (
	"strings"
	"testing"

	"github.com/google/cadvisor/info"
	"github.com/stretchr/testify/mock"
)

type mockContainerHandler struct {
	mock.Mock
}

func (self *mockContainerHandler) GetSpec() (*info.ContainerSpec, error) {
	args := self.Called()
	return args.Get(0).(*info.ContainerSpec), args.Error(1)
}

func (self *mockContainerHandler) ContainerReference() (info.ContainerReference, error) {
	args := self.Called()
	return args.Get(0).(info.ContainerReference), args.Error(1)
}

func (self *mockContainerHandler) GetStats() (*info.ContainerStats, error) {
	args := self.Called()
	return args.Get(0).(*info.ContainerStats), args.Error(1)
}

func (self *mockContainerHandler) ListContainers(listType ListType) ([]info.ContainerReference, error) {
	args := self.Called(listType)
	return args.Get(0).([]info.ContainerReference), args.Error(1)
}

func (self *mockContainerHandler) ListThreads(listType ListType) ([]int, error) {
	args := self.Called(listType)
	return args.Get(0).([]int), args.Error(1)
}

func (self *mockContainerHandler) ListProcesses(listType ListType) ([]int, error) {
	args := self.Called(listType)
	return args.Get(0).([]int), args.Error(1)
}

func TestWhiteListContainerFilter(t *testing.T) {
	mockc := &mockContainerHandler{}
	mockc.On("ListContainers", LIST_RECURSIVE).Return(
		[]info.ContainerReference{
			info.ContainerReference{Name: "/docker/ee0103"},
			info.ContainerReference{Name: "/container/created/by/lmctfy"},
			info.ContainerReference{Name: "/user/something"},
		},
		nil,
	)

	filterPaths := []string{
		"/docker",
		"/container",
	}

	fc := NewWhiteListFilter(mockc, filterPaths...)
	containers, err := fc.ListContainers(LIST_RECURSIVE)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range containers {
		legal := false
		for _, prefix := range filterPaths {
			if strings.HasPrefix(c.Name, prefix) {
				legal = true
			}
		}
		if !legal {
			t.Errorf("%v is not in the white list", c)
		}
	}
	mockc.AssertExpectations(t)
}

func TestBlackListContainerFilter(t *testing.T) {
	mockc := &mockContainerHandler{}
	mockc.On("ListContainers", LIST_RECURSIVE).Return(
		[]info.ContainerReference{
			info.ContainerReference{Name: "/docker/ee0103"},
			info.ContainerReference{Name: "/container/created/by/lmctfy"},
			info.ContainerReference{Name: "/user/something"},
		},
		nil,
	)

	filterPaths := []string{
		"/docker",
		"/container",
	}

	fc := NewBlackListFilter(mockc, filterPaths...)
	containers, err := fc.ListContainers(LIST_RECURSIVE)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range containers {
		legal := true
		for _, prefix := range filterPaths {
			if strings.HasPrefix(c.Name, prefix) {
				legal = false
			}
		}
		if !legal {
			t.Errorf("%v is in the black list", c)
		}
	}
	mockc.AssertExpectations(t)
}
