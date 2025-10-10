/*
 *
 * Copyright 2023 gRPC authors.
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
 *
 */

// Package internal contains functionality internal to the xDS resolver.
package internal

// The following variables are overridden in tests.
var (
	// NewWRR is the function used to create a new weighted round robin
	// implementation.
	NewWRR any // func() wrr.WRR

	// NewXDSClient is the function used to create a new xDS client.
	NewXDSClient any // func(string, estats.MetricsRecorder) (xdsclient.XDSClient, func(), error)
)
