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

package mock

import (
	"github.com/mesos/mesos-go/executor"
	"github.com/mesos/mesos-go/mesosproto"
	"github.com/stretchr/testify/mock"
)

// Executor is used for testing the executor driver.
type Executor struct {
	mock.Mock
}

// New returns a mocked executor.
func New() *Executor {
	return &Executor{}
}

// Registered implements the Registered handler.
func (e *Executor) Registered(executor.ExecutorDriver, *mesosproto.ExecutorInfo, *mesosproto.FrameworkInfo, *mesosproto.SlaveInfo) {
	e.Called()
}

// Reregistered implements the Reregistered handler.
func (e *Executor) Reregistered(executor.ExecutorDriver, *mesosproto.SlaveInfo) {
	e.Called()
}

// Disconnected implements the Disconnected handler.
func (e *Executor) Disconnected(executor.ExecutorDriver) {
	e.Called()
}

// LaunchTask implements the LaunchTask handler.
func (e *Executor) LaunchTask(executor.ExecutorDriver, *mesosproto.TaskInfo) {
	e.Called()
}

// KillTask implements the KillTask handler.
func (e *Executor) KillTask(executor.ExecutorDriver, *mesosproto.TaskID) {
	e.Called()
}

// FrameworkMessage implements the FrameworkMessage handler.
func (e *Executor) FrameworkMessage(executor.ExecutorDriver, string) {
	e.Called()
}

// Shutdown implements the Shutdown handler.
func (e *Executor) Shutdown(executor.ExecutorDriver) {
	e.Called()
}

// Error implements the Error handler.
func (e *Executor) Error(executor.ExecutorDriver, string) {
	e.Called()
}
