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

package fileutil

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestPreallocateExtend(t *testing.T) { runPreallocTest(t, testPreallocateExtend) }
func testPreallocateExtend(t *testing.T, f *os.File) {
	size := int64(64 * 1000)
	if err := Preallocate(f, size, true); err != nil {
		t.Fatal(err)
	}

	stat, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if stat.Size() != size {
		t.Errorf("size = %d, want %d", stat.Size(), size)
	}
}

func TestPreallocateFixed(t *testing.T) { runPreallocTest(t, testPreallocateFixed) }
func testPreallocateFixed(t *testing.T, f *os.File) {
	size := int64(64 * 1000)
	if err := Preallocate(f, size, false); err != nil {
		t.Fatal(err)
	}

	stat, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if stat.Size() != 0 {
		t.Errorf("size = %d, want %d", stat.Size(), 0)
	}
}

func runPreallocTest(t *testing.T, test func(*testing.T, *os.File)) {
	p, err := ioutil.TempDir(os.TempDir(), "preallocateTest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(p)

	f, err := ioutil.TempFile(p, "")
	if err != nil {
		t.Fatal(err)
	}
	test(t, f)
}
