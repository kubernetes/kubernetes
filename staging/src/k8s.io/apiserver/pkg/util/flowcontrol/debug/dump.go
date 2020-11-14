/*
Copyright 2016 The Kubernetes Authors.

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

package debug

import (
	"time"

	"k8s.io/apiserver/pkg/endpoints/request"
)

// QueueSetDump is an instant dump of queue-set.
type QueueSetDump struct {
	Queues    []QueueDump
	Waiting   int
	Executing int
}

// QueueDump is an instant dump of one queue in a queue-set.
type QueueDump struct {
	Requests          []RequestDump
	VirtualStart      float64
	ExecutingRequests int
}

// RequestDump is an instant dump of one requests pending in the queue.
type RequestDump struct {
	MatchedFlowSchema string
	FlowDistinguisher string
	ArriveTime        time.Time
	StartTime         time.Time
	// request details
	UserName    string
	RequestInfo request.RequestInfo
}
