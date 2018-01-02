/*
Copyright 2014 The Camlistore Authors

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

package readerutil

import (
	"os"
	"strings"
	"testing"
)

func TestFakeSeeker(t *testing.T) {
	rs := NewFakeSeeker(strings.NewReader("foobar"), 6)
	if pos, err := rs.Seek(0, os.SEEK_END); err != nil || pos != 6 {
		t.Fatalf("SEEK_END = %d, %v; want 6, nil", pos, err)
	}
	if pos, err := rs.Seek(0, os.SEEK_CUR); err != nil || pos != 6 {
		t.Fatalf("SEEK_CUR = %d, %v; want 6, nil", pos, err)
	}
	if pos, err := rs.Seek(0, os.SEEK_SET); err != nil || pos != 0 {
		t.Fatalf("SEEK_SET = %d, %v; want 0, nil", pos, err)
	}

	buf := make([]byte, 3)
	if n, err := rs.Read(buf); n != 3 || err != nil || string(buf) != "foo" {
		t.Fatalf("First read = %d, %v (buf = %q); want foo", n, err, buf)
	}
	if pos, err := rs.Seek(0, os.SEEK_CUR); err != nil || pos != 3 {
		t.Fatalf("Seek cur pos after first read = %d, %v; want 3, nil", pos, err)
	}
	if n, err := rs.Read(buf); n != 3 || err != nil || string(buf) != "bar" {
		t.Fatalf("Second read = %d, %v (buf = %q); want foo", n, err, buf)
	}

	if pos, err := rs.Seek(1, os.SEEK_SET); err != nil || pos != 1 {
		t.Fatalf("SEEK_SET = %d, %v; want 1, nil", pos, err)
	}
	const msg = "attempt to read from fake seek offset"
	if _, err := rs.Read(buf); err == nil || !strings.Contains(err.Error(), msg) {
		t.Fatalf("bogus Read after seek = %v; want something containing %q", err, msg)
	}
}
