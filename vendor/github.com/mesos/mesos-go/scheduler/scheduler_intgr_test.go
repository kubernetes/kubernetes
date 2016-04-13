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

package scheduler_test

import (
	"io/ioutil"
	"net/http"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	util "github.com/mesos/mesos-go/mesosutil"
	"github.com/mesos/mesos-go/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"

	. "github.com/mesos/mesos-go/scheduler"
)

// testScuduler is used for testing Schduler callbacks.
type testScheduler struct {
	ch     chan bool
	wg     *sync.WaitGroup
	s      *SchedulerIntegrationTestSuite
	errors chan string // yields errors received by Scheduler.Error
}

// convenience
func (sched *testScheduler) T() *testing.T {
	return sched.s.T()
}

func (sched *testScheduler) Registered(dr SchedulerDriver, fw *mesos.FrameworkID, mi *mesos.MasterInfo) {
	log.Infoln("Sched.Registered() called.")
	sched.s.Equal(fw.GetValue(), sched.s.registeredFrameworkId.GetValue(), "driver did not register the expected framework ID")
	sched.s.Equal(mi.GetIp(), uint32(123456))
	sched.ch <- true
}

func (sched *testScheduler) Reregistered(dr SchedulerDriver, mi *mesos.MasterInfo) {
	log.Infoln("Sched.Reregistered() called")
	sched.s.Equal(mi.GetIp(), uint32(123456))
	sched.ch <- true
}

func (sched *testScheduler) Disconnected(dr SchedulerDriver) {
	log.Infoln("Shed.Disconnected() called")
}

func (sched *testScheduler) ResourceOffers(dr SchedulerDriver, offers []*mesos.Offer) {
	log.Infoln("Sched.ResourceOffers called.")
	sched.s.NotNil(offers)
	sched.s.Equal(len(offers), 1)
	sched.ch <- true
}

func (sched *testScheduler) OfferRescinded(dr SchedulerDriver, oid *mesos.OfferID) {
	log.Infoln("Sched.OfferRescinded() called.")
	sched.s.NotNil(oid)
	sched.s.Equal("test-offer-001", oid.GetValue())
	sched.ch <- true
}

func (sched *testScheduler) StatusUpdate(dr SchedulerDriver, stat *mesos.TaskStatus) {
	log.Infoln("Sched.StatusUpdate() called.")
	sched.s.NotNil(stat)
	sched.s.Equal("test-task-001", stat.GetTaskId().GetValue())
	sched.wg.Done()
	log.Infof("Status update done with waitGroup")
}

func (sched *testScheduler) SlaveLost(dr SchedulerDriver, slaveId *mesos.SlaveID) {
	log.Infoln("Sched.SlaveLost() called.")
	sched.s.NotNil(slaveId)
	sched.s.Equal(slaveId.GetValue(), "test-slave-001")
	sched.ch <- true
}

func (sched *testScheduler) FrameworkMessage(dr SchedulerDriver, execId *mesos.ExecutorID, slaveId *mesos.SlaveID, data string) {
	log.Infoln("Sched.FrameworkMessage() called.")
	sched.s.NotNil(slaveId)
	sched.s.Equal(slaveId.GetValue(), "test-slave-001")
	sched.s.NotNil(execId)
	sched.s.NotNil(data)
	sched.s.Equal("test-data-999", string(data))
	sched.ch <- true
}

func (sched *testScheduler) ExecutorLost(SchedulerDriver, *mesos.ExecutorID, *mesos.SlaveID, int) {
	log.Infoln("Sched.ExecutorLost	 called")
}

func (sched *testScheduler) Error(dr SchedulerDriver, err string) {
	log.Infoln("Sched.Error() called.")
	sched.errors <- err
	sched.ch <- true
}

func (sched *testScheduler) waitForCallback(timeout time.Duration) bool {
	if timeout == 0 {
		timeout = 2 * time.Second
	}
	select {
	case <-sched.ch:
		//callback complete
		return true
	case <-time.After(timeout):
		sched.T().Fatalf("timed out after waiting %v for callback", timeout)
	}
	return false
}

func newTestScheduler(s *SchedulerIntegrationTestSuite) *testScheduler {
	return &testScheduler{ch: make(chan bool), s: s, errors: make(chan string, 2)}
}

type mockServerConfigurator func(frameworkId *mesos.FrameworkID, suite *SchedulerIntegrationTestSuite)

type SchedulerIntegrationTestSuiteCore struct {
	SchedulerTestSuiteCore
	server                *testutil.MockMesosHttpServer
	driver                *TestDriver
	sched                 *testScheduler
	config                mockServerConfigurator
	validator             http.HandlerFunc
	registeredFrameworkId *mesos.FrameworkID
}

type SchedulerIntegrationTestSuite struct {
	suite.Suite
	SchedulerIntegrationTestSuiteCore
}

// sets up a mock Mesos HTTP master listener, scheduler, and scheduler driver for testing.
// attempts to wait for a registered or re-registered callback on the suite.sched.
func (suite *SchedulerIntegrationTestSuite) configure(frameworkId *mesos.FrameworkID) bool {
	t := suite.T()
	// start mock master server to handle connection
	suite.server = testutil.NewMockMasterHttpServer(t, func(rsp http.ResponseWriter, req *http.Request) {
		log.Infoln("MockMaster - rcvd ", req.RequestURI)
		if suite.validator != nil {
			suite.validator(rsp, req)
		} else {
			ioutil.ReadAll(req.Body)
			defer req.Body.Close()
			rsp.WriteHeader(http.StatusAccepted)
		}
	})

	t.Logf("test HTTP server listening on %v", suite.server.Addr)
	suite.sched = newTestScheduler(suite)
	suite.sched.ch = make(chan bool, 10) // big enough that it doesn't block callback processing

	cfg := DriverConfig{
		Scheduler: suite.sched,
		Framework: suite.framework,
		Master:    suite.server.Addr,
	}
	suite.driver = newTestDriver(suite.T(), cfg)
	suite.config(frameworkId, suite)

	stat, err := suite.driver.Start()
	suite.NoError(err)
	suite.Equal(mesos.Status_DRIVER_RUNNING, stat)

	ok := waitForConnected(t, suite.driver, 2*time.Second)
	if ok {
		ok = suite.sched.waitForCallback(0) // registered or re-registered callback
	}
	return ok
}

func (suite *SchedulerIntegrationTestSuite) configureServerWithRegisteredFramework() bool {
	// suite.framework is used to initialize the FrameworkInfo of
	// the driver, so if we clear the Id then we'll expect a registration message
	id := suite.framework.Id
	suite.framework.Id = nil
	suite.registeredFrameworkId = id
	return suite.configure(id)
}

var defaultMockServerConfigurator = mockServerConfigurator(func(frameworkId *mesos.FrameworkID, suite *SchedulerIntegrationTestSuite) {
	t := suite.T()
	masterInfo := util.NewMasterInfo("master", 123456, 1234)
	suite.server.On("/master/mesos.internal.RegisterFrameworkMessage").Do(func(rsp http.ResponseWriter, req *http.Request) {
		if suite.validator != nil {
			t.Logf("validating registration request")
			suite.validator(rsp, req)
		} else {
			ioutil.ReadAll(req.Body)
			defer req.Body.Close()
			rsp.WriteHeader(http.StatusAccepted)
		}
		// this is what the mocked scheduler is expecting to receive
		suite.driver.FrameworkRegistered(suite.driver.Context(), suite.driver.MasterPID(), &mesos.FrameworkRegisteredMessage{
			FrameworkId: frameworkId,
			MasterInfo:  masterInfo,
		})
	})
	suite.server.On("/master/mesos.internal.ReregisterFrameworkMessage").Do(func(rsp http.ResponseWriter, req *http.Request) {
		if suite.validator != nil {
			suite.validator(rsp, req)
		} else {
			ioutil.ReadAll(req.Body)
			defer req.Body.Close()
			rsp.WriteHeader(http.StatusAccepted)
		}
		// this is what the mocked scheduler is expecting to receive
		suite.driver.FrameworkReregistered(suite.driver.Context(), suite.driver.MasterPID(), &mesos.FrameworkReregisteredMessage{
			FrameworkId: frameworkId,
			MasterInfo:  masterInfo,
		})
	})
})

func (s *SchedulerIntegrationTestSuite) newMockClient() *testutil.MockMesosClient {
	return testutil.NewMockMesosClient(s.T(), s.server.PID)
}

func (s *SchedulerIntegrationTestSuite) SetupTest() {
	s.SchedulerTestSuiteCore.SetupTest()
	s.config = defaultMockServerConfigurator
}

func (s *SchedulerIntegrationTestSuite) TearDownTest() {
	if s.server != nil {
		s.server.Close()
	}
	if s.driver != nil {
		s.driver.Abort()

		// wait for all events to finish processing, otherwise we can get into a data
		// race when the suite object is reused for the next test.
		<-s.driver.Done()
	}
}

// ---------------------------------- Tests ---------------------------------- //

func TestSchedulerIntegrationSuite(t *testing.T) {
	suite.Run(t, new(SchedulerIntegrationTestSuite))
}

func (suite *SchedulerIntegrationTestSuite) TestSchedulerDriverRegisterFrameworkMessage() {
	t := suite.T()

	id := suite.framework.Id
	suite.framework.Id = nil
	validated := make(chan struct{})
	var closeOnce sync.Once
	suite.validator = http.HandlerFunc(func(rsp http.ResponseWriter, req *http.Request) {
		t.Logf("RCVD request %s", req.URL)

		data, err := ioutil.ReadAll(req.Body)
		if err != nil {
			t.Fatalf("Missing message data from request")
		}
		defer req.Body.Close()

		if "/master/mesos.internal.RegisterFrameworkMessage" != req.RequestURI {
			rsp.WriteHeader(http.StatusAccepted)
			return
		}

		defer closeOnce.Do(func() { close(validated) })

		message := new(mesos.RegisterFrameworkMessage)
		err = proto.Unmarshal(data, message)
		if err != nil {
			t.Fatal("Problem unmarshaling expected RegisterFrameworkMessage")
		}

		suite.NotNil(message)
		info := message.GetFramework()
		suite.NotNil(info)
		suite.Equal(suite.framework.GetName(), info.GetName())
		suite.True(reflect.DeepEqual(suite.framework.GetId(), info.GetId()))
		rsp.WriteHeader(http.StatusOK)
	})
	ok := suite.configure(id)
	suite.True(ok, "failed to establish running test server and driver")
	select {
	case <-time.After(1 * time.Second):
		t.Fatalf("failed to complete validation of framework registration message")
	case <-validated:
		// noop
	}
}

func (suite *SchedulerIntegrationTestSuite) TestSchedulerDriverFrameworkRegisteredEvent() {
	ok := suite.configureServerWithRegisteredFramework()
	suite.True(ok, "failed to establish running test server and driver")
}

func (suite *SchedulerIntegrationTestSuite) TestSchedulerDriverFrameworkReregisteredEvent() {
	ok := suite.configure(suite.framework.Id)
	suite.True(ok, "failed to establish running test server and driver")
}

func (suite *SchedulerIntegrationTestSuite) TestSchedulerDriverResourceOffersEvent() {
	ok := suite.configureServerWithRegisteredFramework()
	suite.True(ok, "failed to establish running test server and driver")

	// Send a event to this SchedulerDriver (via http) to test handlers.
	offer := util.NewOffer(
		util.NewOfferID("test-offer-001"),
		suite.registeredFrameworkId,
		util.NewSlaveID("test-slave-001"),
		"test-localhost",
	)
	pbMsg := &mesos.ResourceOffersMessage{
		Offers: []*mesos.Offer{offer},
		Pids:   []string{"test-offer-001@test-slave-001:5051"},
	}

	c := suite.newMockClient()
	c.SendMessage(suite.driver.UPID(), pbMsg)
	suite.sched.waitForCallback(0)
}

func (suite *SchedulerIntegrationTestSuite) TestSchedulerDriverRescindOfferEvent() {
	ok := suite.configureServerWithRegisteredFramework()
	suite.True(ok, "failed to establish running test server and driver")

	// Send a event to this SchedulerDriver (via http) to test handlers.
	pbMsg := &mesos.RescindResourceOfferMessage{
		OfferId: util.NewOfferID("test-offer-001"),
	}

	c := suite.newMockClient()
	c.SendMessage(suite.driver.UPID(), pbMsg)
	suite.sched.waitForCallback(0)
}

func (suite *SchedulerIntegrationTestSuite) TestSchedulerDriverStatusUpdatedEvent() {
	t := suite.T()
	var wg sync.WaitGroup
	wg.Add(2)
	suite.config = mockServerConfigurator(func(frameworkId *mesos.FrameworkID, suite *SchedulerIntegrationTestSuite) {
		defaultMockServerConfigurator(frameworkId, suite)
		suite.server.On("/master/mesos.internal.StatusUpdateAcknowledgementMessage").Do(func(rsp http.ResponseWriter, req *http.Request) {
			log.Infoln("Master cvd ACK")
			data, _ := ioutil.ReadAll(req.Body)
			defer req.Body.Close()
			assert.NotNil(t, data)
			wg.Done()
			log.Infof("MockMaster - Done with wait group")
		})
		suite.sched.wg = &wg
	})

	ok := suite.configureServerWithRegisteredFramework()
	suite.True(ok, "failed to establish running test server and driver")

	// Send a event to this SchedulerDriver (via http) to test handlers.
	pbMsg := &mesos.StatusUpdateMessage{
		Update: util.NewStatusUpdate(
			suite.registeredFrameworkId,
			util.NewTaskStatus(util.NewTaskID("test-task-001"), mesos.TaskState_TASK_STARTING),
			float64(time.Now().Unix()),
			[]byte("test-abcd-ef-3455-454-001"),
		),
		// note: cannot use driver's pid here if we want an ACK
		Pid: proto.String("test-slave-001(1)@foo.bar:1234"),
	}
	pbMsg.Update.SlaveId = &mesos.SlaveID{Value: proto.String("test-slave-001")}

	c := suite.newMockClient()
	c.SendMessage(suite.driver.UPID(), pbMsg)
	wg.Wait()
}

func (suite *SchedulerIntegrationTestSuite) TestSchedulerDriverLostSlaveEvent() {
	ok := suite.configureServerWithRegisteredFramework()
	suite.True(ok, "failed to establish running test server and driver")

	// Send a event to this SchedulerDriver (via http) to test handlers.	offer := util.NewOffer(
	pbMsg := &mesos.LostSlaveMessage{
		SlaveId: util.NewSlaveID("test-slave-001"),
	}

	c := suite.newMockClient()
	c.SendMessage(suite.driver.UPID(), pbMsg)
	suite.sched.waitForCallback(0)
}

func (suite *SchedulerIntegrationTestSuite) TestSchedulerDriverFrameworkMessageEvent() {
	ok := suite.configureServerWithRegisteredFramework()
	suite.True(ok, "failed to establish running test server and driver")

	// Send a event to this SchedulerDriver (via http) to test handlers.	offer := util.NewOffer(
	pbMsg := &mesos.ExecutorToFrameworkMessage{
		SlaveId:     util.NewSlaveID("test-slave-001"),
		FrameworkId: suite.registeredFrameworkId,
		ExecutorId:  util.NewExecutorID("test-executor-001"),
		Data:        []byte("test-data-999"),
	}

	c := suite.newMockClient()
	c.SendMessage(suite.driver.UPID(), pbMsg)
	suite.sched.waitForCallback(0)
}

func waitForConnected(t *testing.T, driver *TestDriver, timeout time.Duration) bool {
	connected := make(chan struct{})
	go func() {
		defer close(connected)
		for !driver.Connected() {
			time.Sleep(200 * time.Millisecond)
		}
	}()
	select {
	case <-time.After(timeout):
		t.Fatalf("driver failed to establish connection within %v", timeout)
		return false
	case <-connected:
		return true
	}
}

func (suite *SchedulerIntegrationTestSuite) TestSchedulerDriverFrameworkErrorEvent() {
	ok := suite.configureServerWithRegisteredFramework()
	suite.True(ok, "failed to establish running test server and driver")

	// Send an error event to this SchedulerDriver (via http) to test handlers.
	pbMsg := &mesos.FrameworkErrorMessage{
		Message: proto.String("test-error-999"),
	}

	c := suite.newMockClient()
	c.SendMessage(suite.driver.UPID(), pbMsg)
	message := <-suite.sched.errors
	suite.Equal("test-error-999", message)
	suite.sched.waitForCallback(10 * time.Second)
	suite.Equal(mesos.Status_DRIVER_ABORTED, suite.driver.Status())
}
