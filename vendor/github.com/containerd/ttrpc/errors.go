/*
   Copyright The containerd Authors.

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

package ttrpc

import "errors"

var (
	// ErrProtocol is a general error in the handling the protocol.
	ErrProtocol = errors.New("protocol error")

	// ErrClosed is returned by client methods when the underlying connection is
	// closed.
	ErrClosed = errors.New("ttrpc: closed")

	// ErrServerClosed is returned when the Server has closed its connection.
	ErrServerClosed = errors.New("ttrpc: server closed")

	// ErrStreamClosed is when the streaming connection is closed.
	ErrStreamClosed = errors.New("ttrpc: stream closed")
)
