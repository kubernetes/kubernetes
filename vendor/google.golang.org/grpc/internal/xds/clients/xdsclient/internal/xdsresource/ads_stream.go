/*
 *
 * Copyright 2021 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package xdsresource

import "time"

// WatchState is a enum that describes the watch state of a particular
// resource.
type WatchState int

const (
	// ResourceWatchStateStarted is the state where a watch for a resource was
	// started, but a request asking for that resource is yet to be sent to the
	// management server.
	ResourceWatchStateStarted WatchState = iota
	// ResourceWatchStateRequested is the state when a request has been sent for
	// the resource being watched.
	ResourceWatchStateRequested
	// ResourceWatchStateReceived is the state when a response has been received
	// for the resource being watched.
	ResourceWatchStateReceived
	// ResourceWatchStateTimeout is the state when the watch timer associated
	// with the resource expired because no response was received.
	ResourceWatchStateTimeout
)

// ResourceWatchState is the state corresponding to a resource being watched.
type ResourceWatchState struct {
	State       WatchState  // Watch state of the resource.
	ExpiryTimer *time.Timer // Timer for the expiry of the watch.
}
