/*
 *
 * Copyright 2016 gRPC authors.
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

// Package tap defines the function handles which are executed on the transport
// layer of gRPC-Go and related information. Everything here is EXPERIMENTAL.
package tap

import (
	"golang.org/x/net/context"
)

// Info defines the relevant information needed by the handles.
type Info struct {
	// FullMethodName is the string of grpc method (in the format of
	// /package.service/method).
	FullMethodName string
	// TODO: More to be added.
}

// ServerInHandle defines the function which runs when a new stream is created
// on the server side. Note that it is executed in the per-connection I/O goroutine(s) instead
// of per-RPC goroutine. Therefore, users should NOT have any blocking/time-consuming
// work in this handle. Otherwise all the RPCs would slow down.
type ServerInHandle func(ctx context.Context, info *Info) (context.Context, error)
