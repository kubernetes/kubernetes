/*
 *
 * Copyright 2025 gRPC authors.
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

// Package internal contains functionality internal to the xdsclient package.
package internal

import "time"

var (
	// StreamBackoff is the stream backoff for xDS client. It can be overridden
	// by tests to change the default backoff strategy.
	StreamBackoff func(int) time.Duration

	// ResourceWatchStateForTesting gets the watch state for the resource
	// identified by the given resource type and resource name. Returns a
	// non-nil error if there is no such resource being watched.
	ResourceWatchStateForTesting any // func(*xdsclient.XDSClient, xdsclient.ResourceType, string) error
)
