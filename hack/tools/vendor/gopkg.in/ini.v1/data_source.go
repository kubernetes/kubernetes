// Copyright 2019 Unknwon
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package ini

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
)

var (
	_ dataSource = (*sourceFile)(nil)
	_ dataSource = (*sourceData)(nil)
	_ dataSource = (*sourceReadCloser)(nil)
)

// dataSource is an interface that returns object which can be read and closed.
type dataSource interface {
	ReadCloser() (io.ReadCloser, error)
}

// sourceFile represents an object that contains content on the local file system.
type sourceFile struct {
	name string
}

func (s sourceFile) ReadCloser() (_ io.ReadCloser, err error) {
	return os.Open(s.name)
}

// sourceData represents an object that contains content in memory.
type sourceData struct {
	data []byte
}

func (s *sourceData) ReadCloser() (io.ReadCloser, error) {
	return ioutil.NopCloser(bytes.NewReader(s.data)), nil
}

// sourceReadCloser represents an input stream with Close method.
type sourceReadCloser struct {
	reader io.ReadCloser
}

func (s *sourceReadCloser) ReadCloser() (io.ReadCloser, error) {
	return s.reader, nil
}

func parseDataSource(source interface{}) (dataSource, error) {
	switch s := source.(type) {
	case string:
		return sourceFile{s}, nil
	case []byte:
		return &sourceData{s}, nil
	case io.ReadCloser:
		return &sourceReadCloser{s}, nil
	case io.Reader:
		return &sourceReadCloser{ioutil.NopCloser(s)}, nil
	default:
		return nil, fmt.Errorf("error parsing data source: unknown type %q", s)
	}
}
