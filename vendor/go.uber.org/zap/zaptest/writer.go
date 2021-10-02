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

package zaptest

import "go.uber.org/zap/internal/ztest"

type (
	// A Syncer is a spy for the Sync portion of zapcore.WriteSyncer.
	Syncer = ztest.Syncer

	// A Discarder sends all writes to ioutil.Discard.
	Discarder = ztest.Discarder

	// FailWriter is a WriteSyncer that always returns an error on writes.
	FailWriter = ztest.FailWriter

	// ShortWriter is a WriteSyncer whose write method never returns an error,
	// but always reports that it wrote one byte less than the input slice's
	// length (thus, a "short write").
	ShortWriter = ztest.ShortWriter

	// Buffer is an implementation of zapcore.WriteSyncer that sends all writes to
	// a bytes.Buffer. It has convenience methods to split the accumulated buffer
	// on newlines.
	Buffer = ztest.Buffer
)
