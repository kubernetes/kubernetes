// Copyright 2020 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package and provides helpers for adding Close to io.{Reader|Writer}.
package and

import (
	"io"
)

// ReadCloser implements io.ReadCloser by reading from a particular io.Reader
// and then calling the provided "Close()" method.
type ReadCloser struct {
	io.Reader
	CloseFunc func() error
}

var _ io.ReadCloser = (*ReadCloser)(nil)

// Close implements io.ReadCloser
func (rac *ReadCloser) Close() error {
	return rac.CloseFunc()
}

// WriteCloser implements io.WriteCloser by reading from a particular io.Writer
// and then calling the provided "Close()" method.
type WriteCloser struct {
	io.Writer
	CloseFunc func() error
}

var _ io.WriteCloser = (*WriteCloser)(nil)

// Close implements io.WriteCloser
func (wac *WriteCloser) Close() error {
	return wac.CloseFunc()
}
