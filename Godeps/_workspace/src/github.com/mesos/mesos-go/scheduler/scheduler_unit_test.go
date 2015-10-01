/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package scheduler

import (
	"fmt"
	"os/user"
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/detector"
	_ "github.com/mesos/mesos-go/detector/zoo"
	mesos "github.com/mesos/mesos-go/mesosproto"
	util "github.com/mesos/mesos-go/mesosutil"
	"github.com/mesos/mesos-go/messenger"
	"github.com/mesos/mesos-go/upid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
	"golang.org/x/net/context"
)

type SchedulerTestSuiteCore struct {
	master      string
	masterUpid  string
	masterId    string
	frameworkID string
	framework   *mesos.FrameworkInfo
}

type SchedulerTestSuite struct {
	suite.Suite
	SchedulerTestSuiteCore
}

func (s *SchedulerTestSuiteCore) SetupTest() {
	s.master = "127.0.0.1:8080"
	s.masterUpid = "master(2)@" + s.master
	s.masterId = "some-master-id-uuid"
	s.frameworkID = "some-framework-id-uuid"
	s.framework = util.NewFrameworkInfo(
		"test-user",
		"test-name",
		util.NewFrameworkID(s.frameworkID),
	)
}

func TestSchedulerSuite(t *testing.T) {
	t.Logf("running scheduler test suite..")
	suite.Run(t, new(SchedulerTestSuite))
}

func driverConfig(sched Scheduler, framework *mesos.FrameworkInfo, master string, cred *mesos.Credential) DriverConfig {
	return driverConfigMessenger(sched, framework, master, cred, nil)
}

func driverConfigMessenger(sched Scheduler, framework *mesos.FrameworkInfo, master string, cred *mesos.Credential, m messenger.Messenger) DriverConfig {
	d := DriverConfig{
		Scheduler:   sched,
		Framework:   framework,
		Master:      master,
		Credential:  cred,
		NewDetector: func() (detector.Master, error) { return nil, nil }, // master detection not needed
	}
	if m != nil {
		d.NewMessenger = func() (messenger.Messenger, error) { return m, nil }
	}
	return d
}

func mockedMessenger() *messenger.MockedMessenger {
	m := messenger.NewMockedMessenger()
	m.On("Start").Return(nil)
	m.On("UPID").Return(upid.UPID{})
	m.On("Send").Return(nil)
	m.On("Stop").Return(nil)
	m.On("Route").Return(nil)
	m.On("Install").Return(nil)
	return m
}

type testSchedulerDriver struct {
	*MesosSchedulerDriver
}

func (t *testSchedulerDriver) setConnected(b bool) {
	t.eventLock.Lock()
	defer t.eventLock.Unlock()
	t.connected = b
}

func newTestSchedulerDriver(t *testing.T, cfg DriverConfig) *testSchedulerDriver {
	driver, err := NewMesosSchedulerDriver(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return &testSchedulerDriver{driver}
}

func TestSchedulerDriverNew(t *testing.T) {
	masterAddr := "localhost:5050"
	driver := newTestSchedulerDriver(t, driverConfig(NewMockScheduler(), &mesos.FrameworkInfo{}, masterAddr, nil))
	user, _ := user.Current()
	assert.Equal(t, user.Username, driver.frameworkInfo.GetUser())
	host := util.GetHostname("")
	assert.Equal(t, host, driver.frameworkInfo.GetHostname())
}

func TestSchedulerDriverNew_WithPid(t *testing.T) {
	masterAddr := "master@127.0.0.1:5050"
	mUpid, err := upid.Parse(masterAddr)
	assert.NoError(t, err)
	driver := newTestSchedulerDriver(t, driverConfig(NewMockScheduler(), &mesos.FrameworkInfo{}, masterAddr, nil))
	driver.handleMasterChanged(driver.self, &mesos.InternalMasterChangeDetected{Master: &mesos.MasterInfo{Pid: proto.String(mUpid.String())}})
	assert.True(t, driver.masterPid.Equal(mUpid), fmt.Sprintf("expected upid %+v instead of %+v", mUpid, driver.masterPid))
	assert.NoError(t, err)
}

func (suite *SchedulerTestSuite) TestSchedulerDriverNew_WithFrameworkInfo_Override() {
	suite.framework.Hostname = proto.String("local-host")
	driver := newTestSchedulerDriver(suite.T(), driverConfig(NewMockScheduler(), suite.framework, "127.0.0.1:5050", nil))
	suite.Equal(driver.frameworkInfo.GetUser(), "test-user")
	suite.Equal("local-host", driver.frameworkInfo.GetHostname())
}

func (suite *SchedulerTestSuite) TestSchedulerDriverStartOK() {
	sched := NewMockScheduler()
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(sched, suite.framework, suite.master, nil, mockedMessenger()))
	suite.False(driver.Running())

	stat, err := driver.Start()
	suite.NoError(err)
	suite.Equal(mesos.Status_DRIVER_RUNNING, stat)
	suite.True(driver.Running())
	driver.Stop(true)
}

func (suite *SchedulerTestSuite) TestSchedulerDriverStartWithMessengerFailure() {
	sched := NewMockScheduler()
	sched.On("Error").Return()

	messenger := messenger.NewMockedMessenger()
	messenger.On("Start").Return(fmt.Errorf("Failed to start messenger"))
	messenger.On("Stop").Return(nil)
	messenger.On("Install").Return(nil)

	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(sched, suite.framework, suite.master, nil, messenger))
	suite.False(driver.Running())

	stat, err := driver.Start()
	suite.Error(err)
	suite.False(driver.Running())
	suite.False(driver.Connected())
	suite.Equal(mesos.Status_DRIVER_NOT_STARTED, driver.Status())
	suite.Equal(mesos.Status_DRIVER_NOT_STARTED, stat)
}

func (suite *SchedulerTestSuite) TestSchedulerDriverStartWithRegistrationFailure() {
	sched := NewMockScheduler()
	sched.On("Error").Return()

	// Set expections and return values.
	messenger := messenger.NewMockedMessenger()
	messenger.On("Start").Return(nil)
	messenger.On("UPID").Return(upid.UPID{})
	messenger.On("Stop").Return(nil)
	messenger.On("Install").Return(nil)

	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(sched, suite.framework, suite.master, nil, messenger))

	// reliable registration loops until the driver is stopped, connected, etc..
	stat, err := driver.Start()
	suite.NoError(err)
	suite.Equal(mesos.Status_DRIVER_RUNNING, stat)

	time.Sleep(5 * time.Second) // wait a bit, registration should be looping...

	suite.True(driver.Running())
	suite.Equal(mesos.Status_DRIVER_RUNNING, driver.Status())

	// stop the driver, should not panic!
	driver.Stop(false) // intentionally not failing over
	suite.False(driver.Running())
	suite.Equal(mesos.Status_DRIVER_STOPPED, driver.Status())

	messenger.AssertExpectations(suite.T())
}

func (suite *SchedulerTestSuite) TestSchedulerDriverJoinUnstarted() {
	driver := newTestSchedulerDriver(suite.T(), driverConfig(NewMockScheduler(), suite.framework, suite.master, nil))
	suite.False(driver.Running())

	stat, err := driver.Join()
	suite.Error(err)
	suite.Equal(mesos.Status_DRIVER_NOT_STARTED, stat)
	suite.False(driver.Running())
}

func (suite *SchedulerTestSuite) TestSchedulerDriverJoinOK() {
	// Set expections and return values.
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(NewMockScheduler(), suite.framework, suite.master, nil, mockedMessenger()))
	suite.False(driver.Running())

	stat, err := driver.Start()
	suite.NoError(err)
	suite.Equal(mesos.Status_DRIVER_RUNNING, stat)
	suite.True(driver.Running())

	testCh := make(chan mesos.Status)
	go func() {
		stat, _ := driver.Join()
		testCh <- stat
	}()

	driver.Stop(true)
}

func (suite *SchedulerTestSuite) TestSchedulerDriverRun() {
	// Set expections and return values.
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(NewMockScheduler(), suite.framework, suite.master, nil, mockedMessenger()))
	suite.False(driver.Running())

	ch := make(chan struct{})
	go func() {
		defer close(ch)
		stat, err := driver.Run()
		suite.NoError(err)
		suite.Equal(mesos.Status_DRIVER_STOPPED, stat)
	}()
	<-driver.started
	suite.True(driver.Running())
	suite.Equal(mesos.Status_DRIVER_RUNNING, driver.Status())

	// close it all.
	driver.Stop(true)
	<-ch
}

func (suite *SchedulerTestSuite) TestSchedulerDriverStopUnstarted() {
	driver := newTestSchedulerDriver(suite.T(), driverConfig(NewMockScheduler(), suite.framework, suite.master, nil))
	suite.False(driver.Running())

	stat, err := driver.Stop(true)
	suite.NotNil(err)
	suite.False(driver.Running())
	suite.Equal(mesos.Status_DRIVER_NOT_STARTED, stat)
}

type msgTracker struct {
	*messenger.MockedMessenger
	lastMessage proto.Message
}

func (m *msgTracker) Send(ctx context.Context, upid *upid.UPID, msg proto.Message) error {
	m.lastMessage = msg
	return m.MockedMessenger.Send(ctx, upid, msg)
}

func (suite *SchedulerTestSuite) TestSchdulerDriverStop_WithoutFailover() {
	// Set expections and return values.
	messenger := &msgTracker{MockedMessenger: mockedMessenger()}
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(NewMockScheduler(), suite.framework, suite.master, nil, messenger))
	suite.False(driver.Running())

	ch := make(chan struct{})
	go func() {
		defer close(ch)
		stat, err := driver.Run()
		suite.NoError(err)
		suite.Equal(mesos.Status_DRIVER_STOPPED, stat)
	}()
	<-driver.started
	suite.True(driver.Running())
	suite.Equal(mesos.Status_DRIVER_RUNNING, driver.Status())
	driver.connected = true // pretend that we're already registered

	driver.Stop(false)

	msg := messenger.lastMessage
	suite.NotNil(msg)
	_, isUnregMsg := msg.(proto.Message)
	suite.True(isUnregMsg, "expected UnregisterFrameworkMessage instead of %+v", msg)

	suite.False(driver.Running())
	suite.Equal(mesos.Status_DRIVER_STOPPED, driver.Status())
	<-ch
}

func (suite *SchedulerTestSuite) TestSchdulerDriverStop_WithFailover() {
	// Set expections and return values.
	mess := &msgTracker{MockedMessenger: mockedMessenger()}
	d := DriverConfig{
		Scheduler:    NewMockScheduler(),
		Framework:    suite.framework,
		Master:       suite.master,
		NewMessenger: func() (messenger.Messenger, error) { return mess, nil },
		NewDetector:  func() (detector.Master, error) { return nil, nil },
	}
	driver := newTestSchedulerDriver(suite.T(), d)
	suite.False(driver.Running())

	ch := make(chan struct{})
	go func() {
		defer close(ch)
		stat, err := driver.Run()
		suite.NoError(err)
		suite.Equal(mesos.Status_DRIVER_STOPPED, stat)
	}()
	<-driver.started
	driver.setConnected(true) // simulated

	suite.True(driver.Running())
	driver.Stop(true) // true = scheduler failover
	msg := mess.lastMessage

	// we're expecting that lastMessage is nil because when failing over there's no
	// 'unregister' message sent by the scheduler.
	suite.Nil(msg)

	suite.False(driver.Running())
	suite.Equal(mesos.Status_DRIVER_STOPPED, driver.Status())
	<-ch
}

func (suite *SchedulerTestSuite) TestSchedulerDriverAbort() {
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(NewMockScheduler(), suite.framework, suite.master, nil, mockedMessenger()))
	suite.False(driver.Running())

	ch := make(chan struct{})
	go func() {
		defer close(ch)
		stat, err := driver.Run()
		suite.NoError(err)
		suite.Equal(mesos.Status_DRIVER_ABORTED, stat)
	}()
	<-driver.started
	driver.setConnected(true) // simulated

	suite.True(driver.Running())
	suite.Equal(mesos.Status_DRIVER_RUNNING, driver.Status())

	stat, err := driver.Abort()
	suite.NoError(err)

	<-driver.stopCh
	suite.False(driver.Running())
	suite.Equal(mesos.Status_DRIVER_ABORTED, stat)
	suite.Equal(mesos.Status_DRIVER_ABORTED, driver.Status())
	log.Info("waiting for driver to stop")
	<-ch
}

func (suite *SchedulerTestSuite) TestSchdulerDriverLunchTasksUnstarted() {
	sched := NewMockScheduler()
	sched.On("Error").Return()

	// Set expections and return values.
	messenger := messenger.NewMockedMessenger()
	messenger.On("Route").Return(nil)
	messenger.On("Install").Return(nil)

	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(sched, suite.framework, suite.master, nil, messenger))

	stat, err := driver.LaunchTasks(
		[]*mesos.OfferID{{}},
		[]*mesos.TaskInfo{},
		&mesos.Filters{},
	)
	suite.Error(err)
	suite.Equal(mesos.Status_DRIVER_NOT_STARTED, stat)
}

func (suite *SchedulerTestSuite) TestSchdulerDriverLaunchTasksWithError() {
	sched := NewMockScheduler()
	sched.On("StatusUpdate").Return(nil)
	sched.On("Error").Return()

	msgr := mockedMessenger()
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(sched, suite.framework, suite.master, nil, msgr))
	driver.dispatch = func(_ context.Context, _ *upid.UPID, _ proto.Message) error {
		return fmt.Errorf("Unable to send message")
	}

	go func() {
		driver.Run()
	}()
	<-driver.started
	driver.setConnected(true) // simulated
	suite.True(driver.Running())

	// setup an offer
	offer := util.NewOffer(
		util.NewOfferID("test-offer-001"),
		suite.framework.Id,
		util.NewSlaveID("test-slave-001"),
		"test-slave(1)@localhost:5050",
	)

	pid, err := upid.Parse("test-slave(1)@localhost:5050")
	suite.NoError(err)
	driver.cache.putOffer(offer, pid)

	// launch task
	task := util.NewTaskInfo(
		"simple-task",
		util.NewTaskID("simpe-task-1"),
		util.NewSlaveID("test-slave-001"),
		[]*mesos.Resource{util.NewScalarResource("mem", 400)},
	)
	task.Command = util.NewCommandInfo("pwd")
	task.Executor = util.NewExecutorInfo(util.NewExecutorID("test-exec"), task.Command)
	tasks := []*mesos.TaskInfo{task}

	stat, err := driver.LaunchTasks(
		[]*mesos.OfferID{offer.Id},
		tasks,
		&mesos.Filters{},
	)
	suite.Equal(mesos.Status_DRIVER_RUNNING, stat)
	suite.Error(err)
}

func (suite *SchedulerTestSuite) TestSchdulerDriverLaunchTasks() {
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(NewMockScheduler(), suite.framework, suite.master, nil, mockedMessenger()))

	go func() {
		driver.Run()
	}()
	<-driver.started
	driver.setConnected(true) // simulated
	suite.True(driver.Running())

	task := util.NewTaskInfo(
		"simple-task",
		util.NewTaskID("simpe-task-1"),
		util.NewSlaveID("slave-1"),
		[]*mesos.Resource{util.NewScalarResource("mem", 400)},
	)
	task.Command = util.NewCommandInfo("pwd")
	tasks := []*mesos.TaskInfo{task}

	stat, err := driver.LaunchTasks(
		[]*mesos.OfferID{{}},
		tasks,
		&mesos.Filters{},
	)
	suite.NoError(err)
	suite.Equal(mesos.Status_DRIVER_RUNNING, stat)
}

func (suite *SchedulerTestSuite) TestSchdulerDriverKillTask() {
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(NewMockScheduler(), suite.framework, suite.master, nil, mockedMessenger()))

	go func() {
		driver.Run()
	}()
	<-driver.started
	driver.setConnected(true) // simulated
	suite.True(driver.Running())

	stat, err := driver.KillTask(util.NewTaskID("test-task-1"))
	suite.NoError(err)
	suite.Equal(mesos.Status_DRIVER_RUNNING, stat)
}

func (suite *SchedulerTestSuite) TestSchdulerDriverRequestResources() {
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(NewMockScheduler(), suite.framework, suite.master, nil, mockedMessenger()))

	driver.Start()
	driver.setConnected(true) // simulated
	suite.Equal(mesos.Status_DRIVER_RUNNING, driver.Status())

	stat, err := driver.RequestResources(
		[]*mesos.Request{
			{
				SlaveId: util.NewSlaveID("test-slave-001"),
				Resources: []*mesos.Resource{
					util.NewScalarResource("test-res-001", 33.00),
				},
			},
		},
	)
	suite.NoError(err)
	suite.Equal(mesos.Status_DRIVER_RUNNING, stat)
}

func (suite *SchedulerTestSuite) TestSchdulerDriverDeclineOffers() {
	// see LaunchTasks test
}

func (suite *SchedulerTestSuite) TestSchdulerDriverReviveOffers() {
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(NewMockScheduler(), suite.framework, suite.master, nil, mockedMessenger()))

	driver.Start()
	driver.setConnected(true) // simulated
	suite.Equal(mesos.Status_DRIVER_RUNNING, driver.Status())

	stat, err := driver.ReviveOffers()
	suite.NoError(err)
	suite.Equal(mesos.Status_DRIVER_RUNNING, stat)
}

func (suite *SchedulerTestSuite) TestSchdulerDriverSendFrameworkMessage() {
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(NewMockScheduler(), suite.framework, suite.master, nil, mockedMessenger()))

	driver.Start()
	driver.setConnected(true) // simulated
	suite.Equal(mesos.Status_DRIVER_RUNNING, driver.Status())

	stat, err := driver.SendFrameworkMessage(
		util.NewExecutorID("test-exec-001"),
		util.NewSlaveID("test-slave-001"),
		"Hello!",
	)
	suite.NoError(err)
	suite.Equal(mesos.Status_DRIVER_RUNNING, stat)
}

func (suite *SchedulerTestSuite) TestSchdulerDriverReconcileTasks() {
	driver := newTestSchedulerDriver(suite.T(), driverConfigMessenger(NewMockScheduler(), suite.framework, suite.master, nil, mockedMessenger()))

	driver.Start()
	driver.setConnected(true) // simulated
	suite.Equal(mesos.Status_DRIVER_RUNNING, driver.Status())

	stat, err := driver.ReconcileTasks(
		[]*mesos.TaskStatus{
			util.NewTaskStatus(util.NewTaskID("test-task-001"), mesos.TaskState_TASK_FINISHED),
		},
	)
	suite.NoError(err)
	suite.Equal(mesos.Status_DRIVER_RUNNING, stat)
}
