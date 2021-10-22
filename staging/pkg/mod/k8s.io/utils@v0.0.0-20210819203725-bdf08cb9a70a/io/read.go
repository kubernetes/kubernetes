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

package io

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
)

// ErrLimitReached means that the read limit is reached.
var ErrLimitReached = errors.New("the read limit is reached")

// ConsistentRead repeatedly reads a file until it gets the same content twice.
// This is useful when reading files in /proc that are larger than page size
// and kernel may modify them between individual read() syscalls.
// It returns InconsistentReadError when it cannot get a consistent read in
// given nr. of attempts. Caller should retry, kernel is probably under heavy
// mount/unmount load.
func ConsistentRead(filename string, attempts int) ([]byte, error) {
	return consistentReadSync(filename, attempts, nil)
}

// consistentReadSync is the main functionality of ConsistentRead but
// introduces a sync callback that can be used by the tests to mutate the file
// from which the test data is being read
func consistentReadSync(filename string, attempts int, sync func(int)) ([]byte, error) {
	oldContent, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	for i := 0; i < attempts; i++ {
		if sync != nil {
			sync(i)
		}
		newContent, err := ioutil.ReadFile(filename)
		if err != nil {
			return nil, err
		}
		if bytes.Compare(oldContent, newContent) == 0 {
			return newContent, nil
		}
		// Files are different, continue reading
		oldContent = newContent
	}
	return nil, InconsistentReadError{filename, attempts}
}

// InconsistentReadError is returned from ConsistentRead when it cannot get
// a consistent read in given nr. of attempts. Caller should retry, kernel is
// probably under heavy mount/unmount load.
type InconsistentReadError struct {
	filename string
	attempts int
}

func (i InconsistentReadError) Error() string {
	return fmt.Sprintf("could not get consistent content of %s after %d attempts", i.filename, i.attempts)
}

var _ error = InconsistentReadError{}

func IsInconsistentReadError(err error) bool {
	if _, ok := err.(InconsistentReadError); ok {
		return true
	}
	return false
}

// ReadAtMost reads up to `limit` bytes from `r`, and reports an error
// when `limit` bytes are read.
func ReadAtMost(r io.Reader, limit int64) ([]byte, error) {
	limitedReader := &io.LimitedReader{R: r, N: limit}
	data, err := ioutil.ReadAll(limitedReader)
	if err != nil {
		return data, err
	}
	if limitedReader.N <= 0 {
		return data, ErrLimitReached
	}
	return data, nil
}
