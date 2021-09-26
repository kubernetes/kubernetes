/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

func writeToPipe(namedPipe string, flaky bool, i int) {
	pipe, err := os.OpenFile(namedPipe, os.O_WRONLY, 0600)
	if err != nil {
		return
	}

	// The first two reads should never be consistent but all
	// subsequent reads should be
	outstr := fmt.Sprintf("Foobar %t", (i <= 0))

	if flaky {
		outstr = fmt.Sprintf("Foobar %d", i)
	}

	pipe.Write([]byte(outstr))
	pipe.Close()
}

func makePipe(t *testing.T) string {
	tmp, err := ioutil.TempDir("", "pipe-test")
	if err != nil {
		t.Fatal(err)
	}

	pipe := filepath.Join(tmp, "pipe")
	syscall.Mkfifo(pipe, 0600)

	return pipe
}

func writer(namedPipe string, flaky bool, c <-chan int, d <-chan bool) {
	// Make sure something is in the fifo otherwise the first iteration of
	// ConsistentRead will block forever
	writeToPipe(namedPipe, flaky, -1)

	for {
		select {
		case i := <-c:
			writeToPipe(namedPipe, flaky, i)
		case <-d:
			os.RemoveAll(namedPipe)
			return
		}
	}
}

func TestConsistentRead(t *testing.T) {
	pipe := makePipe(t)
	prog, done := make(chan int), make(chan bool)
	go writer(pipe, false, prog, done)

	if _, err := consistentReadSync(pipe, 3, func(i int) { prog <- i }); err != nil {
		t.Fatal(err)
	}

	done <- true
}

func TestConsistentReadFlakyReader(t *testing.T) {
	pipe := makePipe(t)
	prog, done := make(chan int), make(chan bool)
	go writer(pipe, true, prog, done)

	if _, err := consistentReadSync(pipe, 3, func(i int) { prog <- i }); err == nil {
		t.Fatal("flaky reader returned consistent results")
	}
}

func TestReadAtMost(t *testing.T) {
	testCases := []struct {
		limit  int64
		data   string
		errMsg string
	}{
		{4, "hell", "the read limit is reached"},
		{5, "hello", "the read limit is reached"},
		{6, "hello", ""},
	}

	for _, tc := range testCases {
		r := strings.NewReader("hello")
		data, err := ReadAtMost(r, tc.limit)
		if string(data) != tc.data {
			t.Errorf("Read limit %d: expected \"%s\", got \"%s\"", tc.limit, tc.data, string(data))
		}

		if err == nil && tc.errMsg != "" {
			t.Errorf("Read limit %d: expected error with message \"%s\", got no error", tc.limit, tc.errMsg)
		}

		if err != nil && err.Error() != tc.errMsg {
			t.Errorf("Read limit %d: expected error with message \"%s\", got error with message \"%s\"", tc.limit, tc.errMsg, err.Error())
		}
	}
}
