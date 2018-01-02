/*
Copyright 2017 The Kubernetes Authors.

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

package temptest

import (
	"bytes"
	"errors"
	"io"
)

// FakeFile is an implementation of a WriteCloser, that records what has
// been written in the file (in a bytes.Buffer) and if the file has been
// closed.
type FakeFile struct {
	Buffer bytes.Buffer
	Closed bool
}

var _ io.WriteCloser = &FakeFile{}

// Write appends the contents of p to the Buffer. If the file has
// already been closed, an error is returned.
func (f *FakeFile) Write(p []byte) (n int, err error) {
	if f.Closed {
		return 0, errors.New("can't write to closed FakeFile")
	}
	return f.Buffer.Write(p)
}

// Close records that the file has been closed. If the file has already
// been closed, an error is returned.
func (f *FakeFile) Close() error {
	if f.Closed {
		return errors.New("FakeFile was closed multiple times")
	}
	f.Closed = true
	return nil
}
