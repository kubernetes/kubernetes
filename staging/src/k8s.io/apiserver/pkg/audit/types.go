/*
Copyright 2017 The Kubernetes Authors.

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

package audit

import (
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
)

type Sink interface {
	// ProcessEvents handles events. Per audit ID it might be that ProcessEvents is called up to three times.
	// Errors might be logged by the sink itself. If an error should be fatal, leading to an internal
	// error, ProcessEvents is supposed to panic. The event must not be mutated and is reused by the caller
	// after the call returns, i.e. the sink has to make a deepcopy to keep a copy around if necessary.
	ProcessEvents(events ...*auditinternal.Event)
}

type Backend interface {
	Sink

	// Run will initialize the backend. It must not block, but may run go routines in the background. If
	// stopCh is closed, it is supposed to stop them. Run will be called before the first call to ProcessEvents.
	Run(stopCh <-chan struct{}) error

	// Shutdown will synchronously shut down the backend while making sure that all pending
	// events are delivered. It can be assumed that this method is called after
	// the stopCh channel passed to the Run method has been closed.
	Shutdown()

	// Returns the backend PluginName.
	String() string
}
