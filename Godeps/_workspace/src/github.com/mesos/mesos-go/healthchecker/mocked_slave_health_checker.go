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

package healthchecker

import (
	"time"

	"github.com/mesos/mesos-go/upid"
	"github.com/stretchr/testify/mock"
)

type MockedHealthChecker struct {
	mock.Mock
	ch chan time.Time
}

// NewMockedHealthChecker returns a new mocked health checker.
func NewMockedHealthChecker() *MockedHealthChecker {
	return &MockedHealthChecker{ch: make(chan time.Time, 1)}
}

// Start will start the checker and returns the notification channel.
func (m *MockedHealthChecker) Start() <-chan time.Time {
	m.Called()
	return m.ch
}

// Pause will pause the slave health checker.
func (m *MockedHealthChecker) Pause() {
	m.Called()
}

// Continue will continue the slave health checker with a new slave upid.
func (m *MockedHealthChecker) Continue(slaveUPID *upid.UPID) {
	m.Called()
}

// Stop will stop the checker.
func (m *MockedHealthChecker) Stop() {
	m.Called()
}

func (m *MockedHealthChecker) TriggerUnhealthyEvent() {
	m.ch <- time.Now()
}
