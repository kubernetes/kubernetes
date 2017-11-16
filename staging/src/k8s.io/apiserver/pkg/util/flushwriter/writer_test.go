/*
Copyright 2014 The Kubernetes Authors.

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

package flushwriter

import (
	"fmt"
	"testing"
)

type writerWithFlush struct {
	writeCount, flushCount int
	err                    error
}

func (w *writerWithFlush) Flush() {
	w.flushCount++
}

func (w *writerWithFlush) Write(p []byte) (n int, err error) {
	w.writeCount++
	return len(p), w.err
}

type writerWithNoFlush struct {
	writeCount int
}

func (w *writerWithNoFlush) Write(p []byte) (n int, err error) {
	w.writeCount++
	return len(p), nil
}

func TestWriteWithFlush(t *testing.T) {
	w := &writerWithFlush{}
	fw := Wrap(w)
	for i := 0; i < 10; i++ {
		_, err := fw.Write([]byte("Test write"))
		if err != nil {
			t.Errorf("Unexpected error while writing with flush writer: %v", err)
		}
	}
	if w.flushCount != 10 {
		t.Errorf("Flush not called the expected number of times. Actual: %d", w.flushCount)
	}
	if w.writeCount != 10 {
		t.Errorf("Write not called the expected number of times. Actual: %d", w.writeCount)
	}
}

func TestWriteWithoutFlush(t *testing.T) {
	w := &writerWithNoFlush{}
	fw := Wrap(w)
	for i := 0; i < 10; i++ {
		_, err := fw.Write([]byte("Test write"))
		if err != nil {
			t.Errorf("Unexpected error while writing with flush writer: %v", err)
		}
	}
	if w.writeCount != 10 {
		t.Errorf("Write not called the expected number of times. Actual: %d", w.writeCount)
	}
}

func TestWriteError(t *testing.T) {
	e := fmt.Errorf("Error")
	w := &writerWithFlush{err: e}
	fw := Wrap(w)
	_, err := fw.Write([]byte("Test write"))
	if err != e {
		t.Errorf("Did not get expected error. Got: %#v", err)
	}
}
