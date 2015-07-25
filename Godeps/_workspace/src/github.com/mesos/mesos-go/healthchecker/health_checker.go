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
)

// HealthChecker defines the interface of a health checker.
type HealthChecker interface {
	// Start will start the health checker, and returns a notification channel.
	// if the checker thinks the slave is unhealthy, it will send the timestamp
	// via the channel.
	Start() <-chan time.Time
	// Pause will pause the slave health checker.
	Pause()
	// Continue will continue the slave health checker with a new slave upid.
	Continue(slaveUPID *upid.UPID)
	// Stop will stop the health checker. it should be called only once during
	// the life span of the checker.
	Stop()
}
