/*
Copyright 2023 The Kubernetes Authors.

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

package runtime

import (
	"bytes"
	"io"
)

// Splice is the interface that wraps the Splice method.
//
// Splice moves data from given slice without copying the underlying data for
// efficiency purpose. Therefore, the caller should make sure the underlying
// data is not changed later.
type Splice interface {
	Splice([]byte)
	io.Writer
	Reset()
	Bytes() []byte
}

// A spliceBuffer implements Splice and io.Writer interfaces.
type spliceBuffer struct {
	raw []byte
	buf *bytes.Buffer
}

func NewSpliceBuffer() Splice {
	return &spliceBuffer{}
}

// Splice implements the Splice interface.
func (sb *spliceBuffer) Splice(raw []byte) {
	sb.raw = raw
}

// Write implements the io.Writer interface.
func (sb *spliceBuffer) Write(p []byte) (n int, err error) {
	if sb.buf == nil {
		sb.buf = &bytes.Buffer{}
	}
	return sb.buf.Write(p)
}

// Reset resets the buffer to be empty.
func (sb *spliceBuffer) Reset() {
	if sb.buf != nil {
		sb.buf.Reset()
	}
	sb.raw = nil
}

// Bytes returns the data held by the buffer.
func (sb *spliceBuffer) Bytes() []byte {
	if sb.buf != nil && len(sb.buf.Bytes()) > 0 {
		return sb.buf.Bytes()
	}
	if sb.raw != nil {
		return sb.raw
	}
	return []byte{}
}
