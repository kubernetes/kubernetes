// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package ioutil implements I/O utility functions.
package ioutil

import "io"

// NewLimitedBufferReader returns a reader that reads from the given reader
// but limits the amount of data returned to at most n bytes.
func NewLimitedBufferReader(r io.Reader, n int) io.Reader {
	return &limitedBufferReader{
		r: r,
		n: n,
	}
}

type limitedBufferReader struct {
	r io.Reader
	n int
}

func (r *limitedBufferReader) Read(p []byte) (n int, err error) {
	np := p
	if len(np) > r.n {
		np = np[:r.n]
	}
	return r.r.Read(np)
}
