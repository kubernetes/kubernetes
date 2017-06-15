// Copyright (c) 2016 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package transport

import (
	"io"

	"github.com/uber/jaeger-client-go/thrift-gen/zipkincore"
)

// Transport abstracts the method of sending spans out of process.
// Implementations are NOT required to be thread-safe; the RemoteReporter
// is expected to only call methods on the Transport from the same go-routine.
type Transport interface {
	// Append converts the span to the wire representation and adds it
	// to sender's internal buffer.  If the buffer exceeds its designated
	// size, the transport should call Flush() and return the number of spans
	// flushed, otherwise return 0. If error is returned, the returned number
	// of spans is treated as failed span, and reported to metrics accordingly.
	Append(span *zipkincore.Span) (int, error)

	// Flush submits the internal buffer to the remote server. It returns the
	// number of spans flushed. If error is returned, the returned number of
	// spans is treated as failed span, and reported to metrics accordingly.
	Flush() (int, error)

	io.Closer
}
