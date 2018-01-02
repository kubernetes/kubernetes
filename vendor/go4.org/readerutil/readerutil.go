/*
Copyright 2016 The go4 Authors

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

// Package readerutil contains io.Reader types.
package readerutil // import "go4.org/readerutil"

import (
	"expvar"
	"io"
)

// A SizeReaderAt is a ReaderAt with a Size method.
//
// An io.SectionReader implements SizeReaderAt.
type SizeReaderAt interface {
	Size() int64
	io.ReaderAt
}

// A ReadSeekCloser can Read, Seek, and Close.
type ReadSeekCloser interface {
	io.Reader
	io.Seeker
	io.Closer
}

type ReaderAtCloser interface {
	io.ReaderAt
	io.Closer
}

// TODO(wathiede): make sure all the stat readers work with code that
// type asserts ReadFrom/WriteTo.

type varStatReader struct {
	*expvar.Int
	r io.Reader
}

// NewReaderStats returns an io.Reader that will have the number of bytes
// read from r added to v.
func NewStatsReader(v *expvar.Int, r io.Reader) io.Reader {
	return &varStatReader{v, r}
}

func (v *varStatReader) Read(p []byte) (int, error) {
	n, err := v.r.Read(p)
	v.Int.Add(int64(n))
	return n, err
}

type varStatReadSeeker struct {
	*expvar.Int
	rs io.ReadSeeker
}

// NewReaderStats returns an io.ReadSeeker that will have the number of bytes
// read from rs added to v.
func NewStatsReadSeeker(v *expvar.Int, rs io.ReadSeeker) io.ReadSeeker {
	return &varStatReadSeeker{v, rs}
}

func (v *varStatReadSeeker) Read(p []byte) (int, error) {
	n, err := v.rs.Read(p)
	v.Int.Add(int64(n))
	return n, err
}

func (v *varStatReadSeeker) Seek(offset int64, whence int) (int64, error) {
	return v.rs.Seek(offset, whence)
}
