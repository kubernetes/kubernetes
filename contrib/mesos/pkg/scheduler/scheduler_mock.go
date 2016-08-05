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

package scheduler

import (
	"sync"

	"github.com/stretchr/testify/mock"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"time"
)

// MockScheduler implements SchedulerApi
type MockScheduler struct {
	sync.RWMutex
	mock.Mock
}

func (m *MockScheduler) Run(done <-chan struct{}) {
	_ = m.Called()
	runtime.Until(func() {
		time.Sleep(time.Second)
	}, time.Second, done)
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

func (m *MockScheduler) Reconcile(task *podtask.T) {
	_ = m.Called()
	return
}
