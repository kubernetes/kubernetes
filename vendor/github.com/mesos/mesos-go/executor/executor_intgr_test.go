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

package executor

import (
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	util "github.com/mesos/mesos-go/mesosutil"
	"github.com/mesos/mesos-go/testutil"
	"github.com/pborman/uuid"
	"github.com/stretchr/testify/assert"
)

var (
	slavePID    = "slave(1)@127.0.0.1:8080"
	slaveID     = "some-slave-id-uuid"
	frameworkID = "some-framework-id-uuid"
	executorID  = "some-executor-id-uuid"
)

// testScuduler is used for testing Schduler callbacks.
type testExecutor struct {
	ch chan bool
	wg *sync.WaitGroup
	t  *testing.T
}

func newTestExecutor(t *testing.T) *testExecutor {
	return &testExecutor{ch: make(chan bool), t: t}
}

func (exec *testExecutor) Registered(driver ExecutorDriver, execinfo *mesos.ExecutorInfo, fwinfo *mesos.FrameworkInfo, slaveinfo *mesos.SlaveInfo) {
	log.Infoln("Exec.Registered() called.")
	assert.NotNil(exec.t, execinfo)
	assert.NotNil(exec.t, fwinfo)
	assert.NotNil(exec.t, slaveinfo)
	exec.ch <- true
}

func (exec *testExecutor) Reregistered(driver ExecutorDriver, slaveinfo *mesos.SlaveInfo) {
	log.Infoln("Exec.Re-registered() called.")
	assert.NotNil(exec.t, slaveinfo)
	exec.ch <- true
}

func (e *testExecutor) Disconnected(ExecutorDriver) {}

func (exec *testExecutor) LaunchTask(driver ExecutorDriver, taskinfo *mesos.TaskInfo) {
	log.Infoln("Exec.LaunchTask() called.")
	assert.NotNil(exec.t, taskinfo)
	assert.True(exec.t, util.NewTaskID("test-task-001").Equal(taskinfo.TaskId))
	exec.ch <- true
}

func (exec *testExecutor) KillTask(driver ExecutorDriver, taskid *mesos.TaskID) {
	log.Infoln("Exec.KillTask() called.")
	assert.NotNil(exec.t, taskid)
	assert.True(exec.t, util.NewTaskID("test-task-001").Equal(taskid))
	exec.ch <- true
}

func (exec *testExecutor) FrameworkMessage(driver ExecutorDriver, message string) {
	log.Infoln("Exec.FrameworkMessage() called.")
	assert.NotNil(exec.t, message)
	assert.Equal(exec.t, "Hello-Test", message)
	exec.ch <- true
}

func (exec *testExecutor) Shutdown(ExecutorDriver) {
	log.Infoln("Exec.Shutdown() called.")
	exec.ch <- true
}

func (exec *testExecutor) Error(driver ExecutorDriver, err string) {
	log.Infoln("Exec.Error() called.")
	log.Infoln("Got error ", err)
	driver.Stop()
	exec.ch <- true
}

// ------------------------ Test Functions -------------------- //

func setTestEnv(t *testing.T) {
	assert.NoError(t, os.Setenv("MESOS_FRAMEWORK_ID", frameworkID))
	assert.NoError(t, os.Setenv("MESOS_EXECUTOR_ID", executorID))
}

type integrationTestDriver struct {
	*MesosExecutorDriver
}

func (i *integrationTestDriver) setConnected(b bool) {
	i.lock.Lock()
	defer i.lock.Unlock()
	i.connected = b
	i.connection = uuid.NewUUID()
}

// connectionListener returns a signal chan that closes once driver.connected == true.
func (i *integrationTestDriver) connectionListener() chan struct{} {
	connected := make(chan struct{})
	go func() {
		i.lock.Lock()
		defer i.lock.Unlock()
		for !i.connected {
			select {
			case <-i.stopCh:
				return
			default:
				i.cond.Wait()
			}
		}
		close(connected)
	}()
	return connected
}

func newIntegrationTestDriver(t *testing.T, exec Executor) *integrationTestDriver {
	dconfig := DriverConfig{
		Executor: exec,
	}
	driver, err := NewMesosExecutorDriver(dconfig)
	if err != nil {
		t.Fatal(err)
	}
	return &integrationTestDriver{driver}
}

func TestExecutorDriverRegisterExecutorMessage(t *testing.T) {
	setTestEnv(t)
	ch := make(chan bool, 2)
	server := testutil.NewMockSlaveHttpServer(t, func(rsp http.ResponseWriter, req *http.Request) {
		reqPath, err := url.QueryUnescape(req.URL.String())
		assert.NoError(t, err)
		log.Infoln("RCVD request", reqPath)

		data, err := ioutil.ReadAll(req.Body)
		if err != nil {
			t.Fatalf("Missing RegisteredExecutor data from scheduler.")
		}
		defer req.Body.Close()

		message := new(mesos.RegisterExecutorMessage)
		err = proto.Unmarshal(data, message)
		assert.NoError(t, err)
		assert.Equal(t, frameworkID, message.GetFrameworkId().GetValue())
		assert.Equal(t, executorID, message.GetExecutorId().GetValue())

		ch <- true

		rsp.WriteHeader(http.StatusAccepted)
	})

	defer server.Close()

	exec := newTestExecutor(t)
	exec.ch = ch

	driver := newIntegrationTestDriver(t, exec)
	assert.False(t, driver.Running())

	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.True(t, driver.Running())
	assert.Equal(t, mesos.Status_DRIVER_RUNNING, stat)

	select {
	case <-ch:
	case <-time.After(time.Second * 1):
		log.Errorf("Tired of waiting...")
	}
}

func TestExecutorDriverExecutorRegisteredEvent(t *testing.T) {
	setTestEnv(t)
	ch := make(chan bool, 2)
	// Mock Slave process to respond to registration event.
	server := testutil.NewMockSlaveHttpServer(t, func(rsp http.ResponseWriter, req *http.Request) {
		reqPath, err := url.QueryUnescape(req.URL.String())
		assert.NoError(t, err)
		log.Infoln("RCVD request", reqPath)
		rsp.WriteHeader(http.StatusAccepted)
	})

	defer server.Close()

	exec := newTestExecutor(t)
	exec.ch = ch
	exec.t = t

	// start
	driver := newIntegrationTestDriver(t, exec)
	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.Equal(t, mesos.Status_DRIVER_RUNNING, stat)
	defer driver.Stop()

	//simulate sending ExecutorRegisteredMessage from server to exec pid.
	pbMsg := &mesos.ExecutorRegisteredMessage{
		ExecutorInfo:  util.NewExecutorInfo(util.NewExecutorID(executorID), util.NewCommandInfo("ls -l")),
		FrameworkId:   util.NewFrameworkID(frameworkID),
		FrameworkInfo: util.NewFrameworkInfo("test", "test-framework", util.NewFrameworkID(frameworkID)),
		SlaveId:       util.NewSlaveID(slaveID),
		SlaveInfo:     &mesos.SlaveInfo{Hostname: proto.String("localhost")},
	}
	c := testutil.NewMockMesosClient(t, server.PID)
	connected := driver.connectionListener()
	c.SendMessage(driver.self, pbMsg)
	select {
	case <-connected:
	case <-time.After(time.Second * 1):
		log.Errorf("Tired of waiting...")
	}
}

func TestExecutorDriverExecutorReregisteredEvent(t *testing.T) {
	setTestEnv(t)
	ch := make(chan bool, 2)
	// Mock Slave process to respond to registration event.
	server := testutil.NewMockSlaveHttpServer(t, func(rsp http.ResponseWriter, req *http.Request) {
		reqPath, err := url.QueryUnescape(req.URL.String())
		assert.NoError(t, err)
		log.Infoln("RCVD request", reqPath)
		rsp.WriteHeader(http.StatusAccepted)
	})

	defer server.Close()

	exec := newTestExecutor(t)
	exec.ch = ch
	exec.t = t

	// start
	driver := newIntegrationTestDriver(t, exec)
	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.Equal(t, mesos.Status_DRIVER_RUNNING, stat)
	defer driver.Stop()

	//simulate sending ExecutorRegisteredMessage from server to exec pid.
	pbMsg := &mesos.ExecutorReregisteredMessage{
		SlaveId:   util.NewSlaveID(slaveID),
		SlaveInfo: &mesos.SlaveInfo{Hostname: proto.String("localhost")},
	}
	c := testutil.NewMockMesosClient(t, server.PID)
	connected := driver.connectionListener()
	c.SendMessage(driver.self, pbMsg)
	select {
	case <-connected:
	case <-time.After(time.Second * 1):
		log.Errorf("Tired of waiting...")
	}
}

func TestExecutorDriverReconnectEvent(t *testing.T) {
	setTestEnv(t)
	ch := make(chan bool, 2)
	// Mock Slave process to respond to registration event.
	server := testutil.NewMockSlaveHttpServer(t, func(rsp http.ResponseWriter, req *http.Request) {
		reqPath, err := url.QueryUnescape(req.URL.String())
		assert.NoError(t, err)
		log.Infoln("RCVD request", reqPath)

		// exec registration request
		if strings.Contains(reqPath, "RegisterExecutorMessage") {
			log.Infoln("Got Executor registration request")
		}

		if strings.Contains(reqPath, "ReregisterExecutorMessage") {
			log.Infoln("Got Executor Re-registration request")
			ch <- true
		}

		rsp.WriteHeader(http.StatusAccepted)
	})

	defer server.Close()

	exec := newTestExecutor(t)
	exec.t = t

	// start
	driver := newIntegrationTestDriver(t, exec)
	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.Equal(t, mesos.Status_DRIVER_RUNNING, stat)
	driver.setConnected(true)
	defer driver.Stop()

	// send "reconnect" event to driver
	pbMsg := &mesos.ReconnectExecutorMessage{
		SlaveId: util.NewSlaveID(slaveID),
	}
	c := testutil.NewMockMesosClient(t, server.PID)
	c.SendMessage(driver.self, pbMsg)

	select {
	case <-ch:
	case <-time.After(time.Second * 2):
		log.Errorf("Tired of waiting...")
	}

}

func TestExecutorDriverRunTaskEvent(t *testing.T) {
	setTestEnv(t)
	ch := make(chan bool, 2)
	// Mock Slave process to respond to registration event.
	server := testutil.NewMockSlaveHttpServer(t, func(rsp http.ResponseWriter, req *http.Request) {
		reqPath, err := url.QueryUnescape(req.URL.String())
		assert.NoError(t, err)
		log.Infoln("RCVD request", reqPath)
		rsp.WriteHeader(http.StatusAccepted)
	})

	defer server.Close()

	exec := newTestExecutor(t)
	exec.ch = ch
	exec.t = t

	// start
	driver := newIntegrationTestDriver(t, exec)
	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.Equal(t, mesos.Status_DRIVER_RUNNING, stat)
	driver.setConnected(true)
	defer driver.Stop()

	// send runtask event to driver
	pbMsg := &mesos.RunTaskMessage{
		FrameworkId: util.NewFrameworkID(frameworkID),
		Framework: util.NewFrameworkInfo(
			"test", "test-framework-001", util.NewFrameworkID(frameworkID),
		),
		Pid: proto.String(server.PID.String()),
		Task: util.NewTaskInfo(
			"test-task",
			util.NewTaskID("test-task-001"),
			util.NewSlaveID(slaveID),
			[]*mesos.Resource{
				util.NewScalarResource("mem", 112),
				util.NewScalarResource("cpus", 2),
			},
		),
	}

	c := testutil.NewMockMesosClient(t, server.PID)
	c.SendMessage(driver.self, pbMsg)

	select {
	case <-ch:
	case <-time.After(time.Second * 2):
		log.Errorf("Tired of waiting...")
	}

}

func TestExecutorDriverKillTaskEvent(t *testing.T) {
	setTestEnv(t)
	ch := make(chan bool, 2)
	// Mock Slave process to respond to registration event.
	server := testutil.NewMockSlaveHttpServer(t, func(rsp http.ResponseWriter, req *http.Request) {
		reqPath, err := url.QueryUnescape(req.URL.String())
		assert.NoError(t, err)
		log.Infoln("RCVD request", reqPath)
		rsp.WriteHeader(http.StatusAccepted)
	})

	defer server.Close()

	exec := newTestExecutor(t)
	exec.ch = ch
	exec.t = t

	// start
	driver := newIntegrationTestDriver(t, exec)
	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.Equal(t, mesos.Status_DRIVER_RUNNING, stat)
	driver.setConnected(true)
	defer driver.Stop()

	// send runtask event to driver
	pbMsg := &mesos.KillTaskMessage{
		FrameworkId: util.NewFrameworkID(frameworkID),
		TaskId:      util.NewTaskID("test-task-001"),
	}

	c := testutil.NewMockMesosClient(t, server.PID)
	c.SendMessage(driver.self, pbMsg)

	select {
	case <-ch:
	case <-time.After(time.Second * 2):
		log.Errorf("Tired of waiting...")
	}
}

func TestExecutorDriverStatusUpdateAcknowledgement(t *testing.T) {
	setTestEnv(t)
	ch := make(chan bool, 2)
	// Mock Slave process to respond to registration event.
	server := testutil.NewMockSlaveHttpServer(t, func(rsp http.ResponseWriter, req *http.Request) {
		reqPath, err := url.QueryUnescape(req.URL.String())
		assert.NoError(t, err)
		log.Infoln("RCVD request", reqPath)
		rsp.WriteHeader(http.StatusAccepted)
	})

	defer server.Close()

	exec := newTestExecutor(t)
	exec.ch = ch
	exec.t = t

	// start
	driver := newIntegrationTestDriver(t, exec)
	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.Equal(t, mesos.Status_DRIVER_RUNNING, stat)
	driver.setConnected(true)
	defer driver.Stop()

	// send ACK from server
	pbMsg := &mesos.StatusUpdateAcknowledgementMessage{
		SlaveId:     util.NewSlaveID(slaveID),
		FrameworkId: util.NewFrameworkID(frameworkID),
		TaskId:      util.NewTaskID("test-task-001"),
		Uuid:        []byte(uuid.NewRandom().String()),
	}

	c := testutil.NewMockMesosClient(t, server.PID)
	c.SendMessage(driver.self, pbMsg)
	<-time.After(time.Second * 1)
}

func TestExecutorDriverFrameworkToExecutorMessageEvent(t *testing.T) {
	setTestEnv(t)
	ch := make(chan bool, 2)
	// Mock Slave process to respond to registration event.
	server := testutil.NewMockSlaveHttpServer(t, func(rsp http.ResponseWriter, req *http.Request) {
		reqPath, err := url.QueryUnescape(req.URL.String())
		assert.NoError(t, err)
		log.Infoln("RCVD request", reqPath)
		rsp.WriteHeader(http.StatusAccepted)
	})

	defer server.Close()

	exec := newTestExecutor(t)
	exec.ch = ch
	exec.t = t

	// start
	driver := newIntegrationTestDriver(t, exec)
	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.Equal(t, mesos.Status_DRIVER_RUNNING, stat)
	driver.setConnected(true)
	defer driver.Stop()

	// send runtask event to driver
	pbMsg := &mesos.FrameworkToExecutorMessage{
		SlaveId:     util.NewSlaveID(slaveID),
		ExecutorId:  util.NewExecutorID(executorID),
		FrameworkId: util.NewFrameworkID(frameworkID),
		Data:        []byte("Hello-Test"),
	}

	c := testutil.NewMockMesosClient(t, server.PID)
	c.SendMessage(driver.self, pbMsg)

	select {
	case <-ch:
	case <-time.After(time.Second * 1):
		log.Errorf("Tired of waiting...")
	}
}

func TestExecutorDriverShutdownEvent(t *testing.T) {
	setTestEnv(t)
	ch := make(chan bool, 2)
	// Mock Slave process to respond to registration event.
	server := testutil.NewMockSlaveHttpServer(t, func(rsp http.ResponseWriter, req *http.Request) {
		reqPath, err := url.QueryUnescape(req.URL.String())
		assert.NoError(t, err)
		log.Infoln("RCVD request", reqPath)
		rsp.WriteHeader(http.StatusAccepted)
	})

	defer server.Close()

	exec := newTestExecutor(t)
	exec.ch = ch
	exec.t = t

	// start
	driver := newIntegrationTestDriver(t, exec)
	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.Equal(t, mesos.Status_DRIVER_RUNNING, stat)
	driver.setConnected(true)

	// send runtask event to driver
	pbMsg := &mesos.ShutdownExecutorMessage{}

	c := testutil.NewMockMesosClient(t, server.PID)
	c.SendMessage(driver.self, pbMsg)

	select {
	case <-ch:
	case <-time.After(time.Second * 20):
		log.Errorf("Tired of waiting...")
	}

	<-time.After(time.Second * 1) // wait for shutdown to finish.
	assert.Equal(t, mesos.Status_DRIVER_STOPPED, driver.Status())
}
