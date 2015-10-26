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

package types

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/mock"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podschedulers"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/pkg/api"
)

// @deprecated this is a placeholder for me to test the mock package
func TestNoSlavesYet(t *testing.T) {
	obj := &MockScheduler{}
	obj.On("SlaveHostNameFor", "foo").Return(nil)
	obj.SlaveHostNameFor("foo")
	obj.AssertExpectations(t)
}

// MockScheduler implements SchedulerApi
type MockScheduler struct {
	sync.RWMutex
	mock.Mock
}

func (m *MockScheduler) SlaveHostNameFor(id string) (hostName string) {
	args := m.Called(id)
	x := args.Get(0)
	if x != nil {
		hostName = x.(string)
	}
	return
}

func (m *MockScheduler) PodScheduler() (f podschedulers.PodScheduler) {
	args := m.Called()
	x := args.Get(0)
	if x != nil {
		f = x.(podschedulers.PodScheduler)
	}
	return
}

func (m *MockScheduler) CreatePodTask(ctx api.Context, pod *api.Pod) (task *podtask.T, err error) {
	args := m.Called(ctx, pod)
	x := args.Get(0)
	if x != nil {
		task = x.(*podtask.T)
	}
	err = args.Error(1)
	return
}

func (m *MockScheduler) Offers() (f offers.Registry) {
	args := m.Called()
	x := args.Get(0)
	if x != nil {
		f = x.(offers.Registry)
	}
	return
}

func (m *MockScheduler) Tasks() (f podtask.Registry) {
	args := m.Called()
	x := args.Get(0)
	if x != nil {
		f = x.(podtask.Registry)
	}
	return
}

func (m *MockScheduler) KillTask(taskId string) error {
	args := m.Called(taskId)
	return args.Error(0)
}

func (m *MockScheduler) LaunchTask(task *podtask.T) error {
	args := m.Called(task)
	return args.Error(0)
}
