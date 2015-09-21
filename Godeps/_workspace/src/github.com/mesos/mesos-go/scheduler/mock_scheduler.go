package scheduler

import (
	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/stretchr/testify/mock"
)

type MockScheduler struct {
	mock.Mock
}

func NewMockScheduler() *MockScheduler {
	return &MockScheduler{}
}

func (sched *MockScheduler) Registered(SchedulerDriver, *mesos.FrameworkID, *mesos.MasterInfo) {
	sched.Called()
}

func (sched *MockScheduler) Reregistered(SchedulerDriver, *mesos.MasterInfo) {
	sched.Called()
}

func (sched *MockScheduler) Disconnected(SchedulerDriver) {
	sched.Called()
}

func (sched *MockScheduler) ResourceOffers(SchedulerDriver, []*mesos.Offer) {
	sched.Called()
}

func (sched *MockScheduler) OfferRescinded(SchedulerDriver, *mesos.OfferID) {
	sched.Called()
}

func (sched *MockScheduler) StatusUpdate(SchedulerDriver, *mesos.TaskStatus) {
	sched.Called()
}

func (sched *MockScheduler) FrameworkMessage(SchedulerDriver, *mesos.ExecutorID, *mesos.SlaveID, string) {
	sched.Called()
}

func (sched *MockScheduler) SlaveLost(SchedulerDriver, *mesos.SlaveID) {
	sched.Called()
}

func (sched *MockScheduler) ExecutorLost(SchedulerDriver, *mesos.ExecutorID, *mesos.SlaveID, int) {
	sched.Called()
}

func (sched *MockScheduler) Error(d SchedulerDriver, msg string) {
	log.Error(msg)
	sched.Called()
}
