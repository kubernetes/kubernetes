/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package thrift

import (
	"context"
	"errors"
	"io"
)

var errTransportInterrupted = errors.New("Transport Interrupted")

type Flusher interface {
	Flush() (err error)
}

type ContextFlusher interface {
	Flush(ctx context.Context) (err error)
}

type ReadSizeProvider interface {
	RemainingBytes() (num_bytes uint64)
}

// Encapsulates the I/O layer
type TTransport interface {
	io.ReadWriteCloser
	ContextFlusher
	ReadSizeProvider

	// Opens the transport for communication
	Open() error

	// Returns true if the transport is open
	IsOpen() bool
}

type stringWriter interface {
	WriteString(s string) (n int, err error)
}

// This is "enchanced" transport with extra capabilities. You need to use one of these
// to construct protocol.
// Notably, TSocket does not implement this interface, and it is always a mistake to use
// TSocket directly in protocol.
type TRichTransport interface {
	io.ReadWriter
	io.ByteReader
	io.ByteWriter
	stringWriter
	ContextFlusher
	ReadSizeProvider
}
