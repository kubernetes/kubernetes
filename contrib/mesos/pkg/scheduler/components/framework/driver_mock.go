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

package framework

import (
	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/stretchr/testify/mock"
)

/*-----------------------------------------------------------------------------
 |
 |   this really belongs in the mesos-go package, but that's being updated soon
 |   any way so just keep it here for now unless we *really* need it there.
 |
 \-----------------------------------------------------------------------------

// Scheduler defines the interfaces that needed to be implemented.
type Scheduler interface {
        Registered(SchedulerDriver, *FrameworkID, *MasterInfo)
        Reregistered(SchedulerDriver, *MasterInfo)
        Disconnected(SchedulerDriver)
        ResourceOffers(SchedulerDriver, []*Offer)
        OfferRescinded(SchedulerDriver, *OfferID)
        StatusUpdate(SchedulerDriver, *TaskStatus)
        FrameworkMessage(SchedulerDriver, *ExecutorID, *SlaveID, string)
        SlaveLost(SchedulerDriver, *SlaveID)
        ExecutorLost(SchedulerDriver, *ExecutorID, *SlaveID, int)
        Error(SchedulerDriver, string)
}
*/

func status(args mock.Arguments, at int) (val mesos.Status) {
	if x := args.Get(at); x != nil {
		val = x.(mesos.Status)
	}
	return
}

type extendedMock struct {
	mock.Mock
}

// Upon returns a chan that closes upon the execution of the most recently registered call.
func (m *extendedMock) Upon() <-chan struct{} {
	// TODO(jdef) this isn't thread safe, should make it so
	ch := make(chan struct{})
	call := m.ExpectedCalls[len(m.ExpectedCalls)-1]
	f := call.RunFn
	call.RunFn = func(args mock.Arguments) {
		defer close(ch)
		if f != nil {
			f(args)
		}
	}
	return ch
}

type MockSchedulerDriver struct {
	extendedMock
}

func (m *MockSchedulerDriver) Init() error {
	args := m.Called()
	return args.Error(0)
}

func (m *MockSchedulerDriver) Start() (mesos.Status, error) {
	args := m.Called()
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) Stop(b bool) (mesos.Status, error) {
	args := m.Called(b)
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) Abort() (mesos.Status, error) {
	args := m.Called()
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) Join() (mesos.Status, error) {
	args := m.Called()
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) Run() (mesos.Status, error) {
	args := m.Called()
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) RequestResources(r []*mesos.Request) (mesos.Status, error) {
	args := m.Called(r)
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) AcceptOffers(ids []*mesos.OfferID, ops []*mesos.Offer_Operation, f *mesos.Filters) (mesos.Status, error) {
	args := m.Called(ids, ops, f)
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) ReconcileTasks(statuses []*mesos.TaskStatus) (mesos.Status, error) {
	args := m.Called(statuses)
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) LaunchTasks(offerIds []*mesos.OfferID, ti []*mesos.TaskInfo, f *mesos.Filters) (mesos.Status, error) {
	args := m.Called(offerIds, ti, f)
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) KillTask(tid *mesos.TaskID) (mesos.Status, error) {
	args := m.Called(tid)
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) DeclineOffer(oid *mesos.OfferID, f *mesos.Filters) (mesos.Status, error) {
	args := m.Called(oid, f)
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) ReviveOffers() (mesos.Status, error) {
	args := m.Called()
	return status(args, 0), args.Error(0)
}

func (m *MockSchedulerDriver) SendFrameworkMessage(eid *mesos.ExecutorID, sid *mesos.SlaveID, s string) (mesos.Status, error) {
	args := m.Called(eid, sid, s)
	return status(args, 0), args.Error(1)
}

func (m *MockSchedulerDriver) Destroy() {
	m.Called()
}

func (m *MockSchedulerDriver) Wait() {
	m.Called()
}

type JoinableDriver struct {
	MockSchedulerDriver
	joinFunc func() (mesos.Status, error)
}

// Join invokes joinFunc if it has been set, otherwise blocks forever
func (m *JoinableDriver) Join() (mesos.Status, error) {
	if m.joinFunc != nil {
		return m.joinFunc()
	}
	select {}
}
