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

package executor_test

import (
	"fmt"
	"os"
	"testing"

	mock_executor "github.com/mesos/mesos-go/executor/mock"
	"github.com/mesos/mesos-go/mesosproto"
	util "github.com/mesos/mesos-go/mesosutil"
	mock_messenger "github.com/mesos/mesos-go/messenger/mock"
	"github.com/mesos/mesos-go/upid"
	"github.com/stretchr/testify/assert"

	. "github.com/mesos/mesos-go/executor"
)

var (
	slavePID    = "slave(1)@127.0.0.1:8080"
	slaveID     = "some-slave-id-uuid"
	frameworkID = "some-framework-id-uuid"
	executorID  = "some-executor-id-uuid"
)

func setEnvironments(t *testing.T, workDir string, checkpoint bool) {
	assert.NoError(t, os.Setenv("MESOS_SLAVE_PID", slavePID))
	assert.NoError(t, os.Setenv("MESOS_SLAVE_ID", slaveID))
	assert.NoError(t, os.Setenv("MESOS_FRAMEWORK_ID", frameworkID))
	assert.NoError(t, os.Setenv("MESOS_EXECUTOR_ID", executorID))
	if len(workDir) > 0 {
		assert.NoError(t, os.Setenv("MESOS_DIRECTORY", workDir))
	}
	if checkpoint {
		assert.NoError(t, os.Setenv("MESOS_CHECKPOINT", "1"))
	}
}

func clearEnvironments(t *testing.T) {
	assert.NoError(t, os.Setenv("MESOS_SLAVE_PID", ""))
	assert.NoError(t, os.Setenv("MESOS_SLAVE_ID", ""))
	assert.NoError(t, os.Setenv("MESOS_FRAMEWORK_ID", ""))
	assert.NoError(t, os.Setenv("MESOS_EXECUTOR_ID", ""))
}

func newTestExecutorDriver(t *testing.T, exec Executor) *TestDriver {
	dconfig := DriverConfig{
		Executor: exec,
	}
	driver, err := NewMesosExecutorDriver(dconfig)
	if err != nil {
		t.Fatal(err)
	}
	return &TestDriver{driver}
}

func createTestExecutorDriver(t *testing.T) (
	*TestDriver,
	*mock_messenger.Messenger) {

	exec := mock_executor.New()
	exec.On("Error").Return(nil)

	setEnvironments(t, "", false)
	driver := newTestExecutorDriver(t, exec)

	messenger := mock_messenger.NewMessenger()
	messenger.On("Start").Return(nil)
	messenger.On("UPID").Return(upid.UPID{})
	messenger.On("Send").Return(nil)
	messenger.On("Stop").Return(nil)

	driver.SetMessenger(messenger)
	return driver, messenger
}

func TestExecutorDriverStartFailedToParseEnvironment(t *testing.T) {
	clearEnvironments(t)
	exec := mock_executor.New()
	exec.On("Error").Return(nil)
	dconfig := DriverConfig{Executor: exec}
	driver, err := NewMesosExecutorDriver(dconfig)
	assert.Nil(t, driver)
	assert.Error(t, err)
}

func TestExecutorDriverStartFailedToStartMessenger(t *testing.T) {
	exec := mock_executor.New()

	setEnvironments(t, "", false)
	driver := newTestExecutorDriver(t, exec)
	assert.NotNil(t, driver)
	messenger := mock_messenger.NewMessenger()
	driver.SetMessenger(messenger)

	// Set expections and return values.
	messenger.On("Start").Return(fmt.Errorf("messenger failed to start"))
	messenger.On("Stop").Return(nil)

	status, err := driver.Start()
	assert.Error(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_NOT_STARTED, status)

	messenger.Stop()

	messenger.AssertNumberOfCalls(t, "Start", 1)
	messenger.AssertNumberOfCalls(t, "Stop", 1)
}

func TestExecutorDriverStartFailedToSendRegisterMessage(t *testing.T) {
	exec := mock_executor.New()

	setEnvironments(t, "", false)
	driver := newTestExecutorDriver(t, exec)
	messenger := mock_messenger.NewMessenger()
	driver.SetMessenger(messenger)

	// Set expections and return values.
	messenger.On("Start").Return(nil)
	messenger.On("UPID").Return(upid.UPID{})
	messenger.On("Send").Return(fmt.Errorf("messenger failed to send"))
	messenger.On("Stop").Return(nil)

	status, err := driver.Start()
	assert.Error(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_NOT_STARTED, status)

	messenger.AssertNumberOfCalls(t, "Start", 1)
	messenger.AssertNumberOfCalls(t, "UPID", 1)
	messenger.AssertNumberOfCalls(t, "Send", 1)
	messenger.AssertNumberOfCalls(t, "Stop", 1)
}

func TestExecutorDriverStartSucceed(t *testing.T) {
	setEnvironments(t, "", false)

	exec := mock_executor.New()
	exec.On("Error").Return(nil)

	driver := newTestExecutorDriver(t, exec)

	messenger := mock_messenger.NewMessenger()
	driver.SetMessenger(messenger)
	messenger.On("Start").Return(nil)
	messenger.On("UPID").Return(upid.UPID{})
	messenger.On("Send").Return(nil)
	messenger.On("Stop").Return(nil)

	assert.False(t, driver.Running())
	status, err := driver.Start()
	assert.True(t, driver.Running())
	assert.NoError(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_RUNNING, status)

	messenger.AssertNumberOfCalls(t, "Start", 1)
	messenger.AssertNumberOfCalls(t, "UPID", 1)
	messenger.AssertNumberOfCalls(t, "Send", 1)
}

func TestExecutorDriverRun(t *testing.T) {
	setEnvironments(t, "", false)

	// Set expections and return values.
	messenger := mock_messenger.NewMessenger()
	messenger.On("Start").Return(nil)
	messenger.On("UPID").Return(upid.UPID{})
	messenger.On("Send").Return(nil)
	messenger.On("Stop").Return(nil)

	exec := mock_executor.New()
	exec.On("Error").Return(nil)

	driver := newTestExecutorDriver(t, exec)
	driver.SetMessenger(messenger)
	assert.False(t, driver.Running())

	ch := make(chan struct{})
	go func() {
		defer close(ch)
		stat, err := driver.Run()
		assert.NoError(t, err)
		assert.Equal(t, mesosproto.Status_DRIVER_STOPPED, stat)
	}()
	<-driver.Started()
	assert.True(t, driver.Running())
	driver.Stop()
	<-ch
}

func TestExecutorDriverJoin(t *testing.T) {
	setEnvironments(t, "", false)

	// Set expections and return values.
	messenger := mock_messenger.NewMessenger()
	messenger.On("Start").Return(nil)
	messenger.On("UPID").Return(upid.UPID{})
	messenger.On("Send").Return(nil)
	messenger.On("Stop").Return(nil)

	exec := mock_executor.New()
	exec.On("Error").Return(nil)

	driver := newTestExecutorDriver(t, exec)
	driver.SetMessenger(messenger)
	assert.False(t, driver.Running())

	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.True(t, driver.Running())
	assert.Equal(t, mesosproto.Status_DRIVER_RUNNING, stat)

	testCh := make(chan mesosproto.Status)
	go func() {
		stat, _ := driver.Join()
		testCh <- stat
	}()

	driver.Stop()
	stat = <-testCh // when Stop() is called, stat will be DRIVER_STOPPED.
	assert.Equal(t, mesosproto.Status_DRIVER_STOPPED, stat)
}

func TestExecutorDriverAbort(t *testing.T) {
	statusChan := make(chan mesosproto.Status)
	driver, messenger := createTestExecutorDriver(t)

	assert.False(t, driver.Running())
	stat, err := driver.Start()
	assert.True(t, driver.Running())
	assert.NoError(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_RUNNING, stat)
	go func() {
		st, _ := driver.Join()
		statusChan <- st
	}()

	stat, err = driver.Abort()
	assert.NoError(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_ABORTED, stat)
	assert.Equal(t, mesosproto.Status_DRIVER_ABORTED, <-statusChan)
	assert.False(t, driver.Running())

	// Abort for the second time, should return directly.
	stat, err = driver.Abort()
	assert.Error(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_ABORTED, stat)
	stat, err = driver.Stop()
	assert.Error(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_ABORTED, stat)
	assert.False(t, driver.Running())

	// Restart should not start.
	stat, err = driver.Start()
	assert.False(t, driver.Running())
	assert.Error(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_ABORTED, stat)

	messenger.AssertNumberOfCalls(t, "Start", 1)
	messenger.AssertNumberOfCalls(t, "UPID", 1)
	messenger.AssertNumberOfCalls(t, "Send", 1)
	messenger.AssertNumberOfCalls(t, "Stop", 1)
}

func TestExecutorDriverStop(t *testing.T) {
	statusChan := make(chan mesosproto.Status)
	driver, messenger := createTestExecutorDriver(t)

	assert.False(t, driver.Running())
	stat, err := driver.Start()
	assert.True(t, driver.Running())
	assert.NoError(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_RUNNING, stat)
	go func() {
		stat, _ := driver.Join()
		statusChan <- stat
	}()
	stat, err = driver.Stop()
	assert.NoError(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_STOPPED, stat)
	assert.Equal(t, mesosproto.Status_DRIVER_STOPPED, <-statusChan)
	assert.False(t, driver.Running())

	// Stop for the second time, should return directly.
	stat, err = driver.Stop()
	assert.Error(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_STOPPED, stat)
	stat, err = driver.Abort()
	assert.Error(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_STOPPED, stat)
	assert.False(t, driver.Running())

	// Restart should not start.
	stat, err = driver.Start()
	assert.False(t, driver.Running())
	assert.Error(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_STOPPED, stat)

	messenger.AssertNumberOfCalls(t, "Start", 1)
	messenger.AssertNumberOfCalls(t, "UPID", 1)
	messenger.AssertNumberOfCalls(t, "Send", 1)
	messenger.AssertNumberOfCalls(t, "Stop", 1)
}

func TestExecutorDriverSendStatusUpdate(t *testing.T) {

	driver, _ := createTestExecutorDriver(t)

	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_RUNNING, stat)
	driver.SetConnected(true)

	taskStatus := util.NewTaskStatus(
		util.NewTaskID("test-task-001"),
		mesosproto.TaskState_TASK_RUNNING,
	)

	stat, err = driver.SendStatusUpdate(taskStatus)
	assert.NoError(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_RUNNING, stat)
}

func TestExecutorDriverSendStatusUpdateStaging(t *testing.T) {

	driver, _ := createTestExecutorDriver(t)
	stat, err := driver.Start()
	assert.NoError(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_RUNNING, stat)
	driver.SetConnected(true)

	taskStatus := util.NewTaskStatus(
		util.NewTaskID("test-task-001"),
		mesosproto.TaskState_TASK_STAGING,
	)

	stat, err = driver.SendStatusUpdate(taskStatus)
	assert.Error(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_ABORTED, stat)
}

func TestExecutorDriverSendFrameworkMessage(t *testing.T) {

	driver, _ := createTestExecutorDriver(t)

	stat, err := driver.SendFrameworkMessage("failed")
	assert.Error(t, err)

	stat, err = driver.Start()
	assert.NoError(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_RUNNING, stat)
	driver.SetConnected(true)

	stat, err = driver.SendFrameworkMessage("Testing Mesos")
	assert.NoError(t, err)
	assert.Equal(t, mesosproto.Status_DRIVER_RUNNING, stat)
}

func TestStatusUpdateAckRace_Issue103(t *testing.T) {
	driver, _ := createTestExecutorDriver(t)
	_, err := driver.Start()
	assert.NoError(t, err)

	msg := &mesosproto.StatusUpdateAcknowledgementMessage{}
	go driver.StatusUpdateAcknowledgement(driver.Context(), nil, msg)

	taskStatus := util.NewTaskStatus(
		util.NewTaskID("test-task-001"),
		mesosproto.TaskState_TASK_STAGING,
	)

	driver.SendStatusUpdate(taskStatus)
}
