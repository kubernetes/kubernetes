/*
Copyright 2021 The Kubernetes Authors.

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

package server

/*
We make an attempt here to identify the events that take place during
lifecycle of the apiserver.

We also identify each event with a name so we can refer to it.

Events:
- ShutdownInitiated: KILL signal received
- AfterShutdownDelayDuration: shutdown delay duration has passed
- InFlightRequestsDrained: all in flight request(s) have been drained
- HasBeenReady is signaled when the readyz endpoint succeeds for the first time

The following is a sequence of shutdown events that we expect to see during termination:
T0: ShutdownInitiated: KILL signal received
	- /readyz starts returning red
    - run pre shutdown hooks

T0+70s: AfterShutdownDelayDuration: shutdown delay duration has passed
	- the default value of 'ShutdownDelayDuration' is '70s'
	- it's time to initiate shutdown of the HTTP Server, server.Shutdown is invoked
	- as a consequene, the Close function has is called for all listeners
 	- the HTTP Server stops listening immediately
	- any new request arriving on a new TCP socket is denied with
      a network error similar to 'connection refused'
    - the HTTP Server waits gracefully for existing requests to complete
      up to '60s' (dictated by ShutdownTimeout)
	- active long running requests will receive a GOAWAY.

T0+70s: HTTPServerStoppedListening:
	- this event is signaled when the HTTP Server has stopped listening
      which is immediately after server.Shutdown has been invoked

T0 + 70s + up-to 60s: InFlightRequestsDrained: existing in flight requests have been drained
	- long running requests are outside of this scope
	- up-to 60s: the default value of 'ShutdownTimeout' is 60s, this means that
      any request in flight has a hard timeout of 60s.
	- it's time to call 'Shutdown' on the audit events since all
	  in flight request(s) have drained.
*/

// lifecycleSignal encapsulates a named apiserver event
type lifecycleSignal interface {
	// Signal signals the event, indicating that the event has occurred.
	// Signal is idempotent, once signaled the event stays signaled and
	// it immediately unblocks any goroutine waiting for this event.
	Signal()

	// Signaled returns a channel that is closed when the underlying event
	// has been signaled. Successive calls to Signaled return the same value.
	Signaled() <-chan struct{}

	// Name returns the name of the signal, useful for logging.
	Name() string
}

// lifecycleSignals provides an abstraction of the events that
// transpire during the lifecycle of the apiserver. This abstraction makes it easy
// for us to write unit tests that can verify expected graceful termination behavior.
//
// GenericAPIServer can use these to either:
//  - signal that a particular termination event has transpired
//  - wait for a designated termination event to transpire and do some action.
type lifecycleSignals struct {
	// ShutdownInitiated event is signaled when an apiserver shutdown has been initiated.
	// It is signaled when the `stopCh` provided by the main goroutine
	// receives a KILL signal and is closed as a consequence.
	ShutdownInitiated lifecycleSignal

	// AfterShutdownDelayDuration event is signaled as soon as ShutdownDelayDuration
	// has elapsed since the ShutdownInitiated event.
	// ShutdownDelayDuration allows the apiserver to delay shutdown for some time.
	AfterShutdownDelayDuration lifecycleSignal

	// InFlightRequestsDrained event is signaled when the existing requests
	// in flight have completed. This is used as signal to shut down the audit backends
	InFlightRequestsDrained lifecycleSignal

	// HTTPServerStoppedListening termination event is signaled when the
	// HTTP Server has stopped listening to the underlying socket.
	HTTPServerStoppedListening lifecycleSignal

	// HasBeenReady is signaled when the readyz endpoint succeeds for the first time.
	HasBeenReady lifecycleSignal
}

// newLifecycleSignals returns an instance of lifecycleSignals interface to be used
// to coordinate lifecycle of the apiserver
func newLifecycleSignals() lifecycleSignals {
	return lifecycleSignals{
		ShutdownInitiated:          newNamedChannelWrapper("ShutdownInitiated"),
		AfterShutdownDelayDuration: newNamedChannelWrapper("AfterShutdownDelayDuration"),
		InFlightRequestsDrained:    newNamedChannelWrapper("InFlightRequestsDrained"),
		HTTPServerStoppedListening: newNamedChannelWrapper("HTTPServerStoppedListening"),
		HasBeenReady:               newNamedChannelWrapper("HasBeenReady"),
	}
}

func newNamedChannelWrapper(name string) lifecycleSignal {
	return &namedChannelWrapper{
		name: name,
		ch:   make(chan struct{}),
	}
}

type namedChannelWrapper struct {
	name string
	ch   chan struct{}
}

func (e *namedChannelWrapper) Signal() {
	select {
	case <-e.ch:
		// already closed, don't close again.
	default:
		close(e.ch)
	}
}

func (e *namedChannelWrapper) Signaled() <-chan struct{} {
	return e.ch
}

func (e *namedChannelWrapper) Name() string {
	return e.name
}
