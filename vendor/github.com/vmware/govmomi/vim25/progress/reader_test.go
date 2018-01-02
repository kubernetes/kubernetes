/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package progress

import (
	"io"
	"strings"
	"testing"
)

func TestReader(t *testing.T) {
	s := "helloworld"
	ch := make(chan Report, 1)
	pr := NewReader(&dummySinker{ch}, strings.NewReader(s), int64(len(s)))

	var buf [10]byte
	var q Report
	var n int
	var err error

	// Read first byte
	n, err = pr.Read(buf[0:1])
	if n != 1 {
		t.Errorf("Expected n=1, but got: %d", n)
	}

	if err != nil {
		t.Errorf("Error: %s", err)
	}

	q = <-ch
	if q.Error() != nil {
		t.Errorf("Error: %s", err)
	}

	if f := q.Percentage(); f != 10.0 {
		t.Errorf("Expected percentage after 1 byte to be 10%%, but got: %.0f%%", f)
	}

	// Read remaining bytes
	n, err = pr.Read(buf[:])
	if n != 9 {
		t.Errorf("Expected n=1, but got: %d", n)
	}
	if err != nil {
		t.Errorf("Error: %s", err)
	}

	q = <-ch
	if q.Error() != nil {
		t.Errorf("Error: %s", err)
	}

	if f := q.Percentage(); f != 100.0 {
		t.Errorf("Expected percentage after 10 bytes to be 100%%, but got: %.0f%%", f)
	}

	// Read EOF
	_, err = pr.Read(buf[:])
	if err != io.EOF {
		t.Errorf("Expected io.EOF, but got: %s", err)
	}

	// Mark progress reader as done
	pr.Done(io.EOF)
	q = <-ch
	if err != io.EOF {
		t.Errorf("Expected io.EOF, but got: %s", err)
	}

	// Progress channel should be closed after progress reader is marked done
	_, ok := <-ch
	if ok {
		t.Errorf("Expected channel to be closed")
	}
}
