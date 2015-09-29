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

package messenger

import (
	"github.com/mesos/mesos-go/upid"
	"golang.org/x/net/context"
)

// Transporter defines methods for communicating with remote processes.
type Transporter interface {
	//Send sends message to remote process. Must use context to determine
	//cancelled requests. Will stop sending when transport is stopped.
	Send(ctx context.Context, msg *Message) error

	//Rcvd receives and delegate message handling to installed handlers.
	//Will stop receiving when transport is stopped.
	Recv() (*Message, error)

	//Inject injects a message to the incoming queue. Must use context to
	//determine cancelled requests. Injection is aborted if the transport
	//is stopped.
	Inject(ctx context.Context, msg *Message) error

	//Install mount an handler based on incoming message name.
	Install(messageName string)

	//Start starts the transporter and returns immediately. The error chan
	//is never nil.
	Start() (upid.UPID, <-chan error)

	//Stop kills the transporter.
	Stop(graceful bool) error

	//UPID returns the PID for transporter.
	UPID() upid.UPID
}
