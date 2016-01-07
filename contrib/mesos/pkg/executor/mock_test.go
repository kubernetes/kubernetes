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

package executor

import (
	"testing"

	"github.com/mesos/mesos-go/mesosproto"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

type MockExecutorDriver struct {
	mock.Mock
}

func (m *MockExecutorDriver) Start() (mesosproto.Status, error) {
	args := m.Called()
	return args.Get(0).(mesosproto.Status), args.Error(1)
}

func (m *MockExecutorDriver) Stop() (mesosproto.Status, error) {
	args := m.Called()
	return args.Get(0).(mesosproto.Status), args.Error(1)
}

func (m *MockExecutorDriver) Abort() (mesosproto.Status, error) {
	args := m.Called()
	return args.Get(0).(mesosproto.Status), args.Error(1)
}

func (m *MockExecutorDriver) Join() (mesosproto.Status, error) {
	args := m.Called()
	return args.Get(0).(mesosproto.Status), args.Error(1)
}

func (m *MockExecutorDriver) Run() (mesosproto.Status, error) {
	args := m.Called()
	return args.Get(0).(mesosproto.Status), args.Error(1)
}

func (m *MockExecutorDriver) SendStatusUpdate(taskStatus *mesosproto.TaskStatus) (mesosproto.Status, error) {
	args := m.Called(*taskStatus.State)
	return args.Get(0).(mesosproto.Status), args.Error(1)
}

func (m *MockExecutorDriver) SendFrameworkMessage(msg string) (mesosproto.Status, error) {
	args := m.Called(msg)
	return args.Get(0).(mesosproto.Status), args.Error(1)
}

func NewTestKubernetesExecutor() (*Executor, chan kubetypes.PodUpdate) {
	updates := make(chan kubetypes.PodUpdate, 1024)
	return New(Config{
		Docker:  dockertools.ConnectToDockerOrDie("fake://"),
		Updates: updates,
		PodLW:   &NewMockPodsListWatch(api.PodList{}).ListWatch,
	}), updates
}

func TestExecutorNew(t *testing.T) {
	mockDriver := &MockExecutorDriver{}
	executor, _ := NewTestKubernetesExecutor()
	executor.Init(mockDriver)

	assert.Equal(t, executor.isDone(), false, "executor should not be in Done state on initialization")
	assert.Equal(t, executor.isConnected(), false, "executor should not be connected on initialization")
}
