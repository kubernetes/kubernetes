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
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"testing"
	"time"
)

func TestPurgeFile(t *testing.T) {
	dir, err := ioutil.TempDir("", "purgefile")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	// minimal file set
	for i := 0; i < 3; i++ {
		f, ferr := os.Create(path.Join(dir, fmt.Sprintf("%d.test", i)))
		if ferr != nil {
			t.Fatal(err)
		}
		f.Close()
	}

	stop, purgec := make(chan struct{}), make(chan string, 10)

	// keep 3 most recent files
	errch := purgeFile(dir, "test", 3, time.Millisecond, stop, purgec)
	select {
	case f := <-purgec:
		t.Errorf("unexpected purge on %q", f)
	case <-time.After(10 * time.Millisecond):
	}

	// rest of the files
	for i := 4; i < 10; i++ {
		go func(n int) {
			f, ferr := os.Create(path.Join(dir, fmt.Sprintf("%d.test", n)))
			if ferr != nil {
				t.Fatal(err)
			}
			f.Close()
		}(i)
	}

	// watch files purge away
	for i := 4; i < 10; i++ {
		select {
		case <-purgec:
		case <-time.After(time.Second):
			t.Errorf("purge took too long")
		}
	}

	fnames, rerr := ReadDir(dir)
	if rerr != nil {
		t.Fatal(rerr)
	}
	wnames := []string{"7.test", "8.test", "9.test"}
	if !reflect.DeepEqual(fnames, wnames) {
		t.Errorf("filenames = %v, want %v", fnames, wnames)
	}

	// no error should be reported from purge routine
	select {
	case f := <-purgec:
		t.Errorf("unexpected purge on %q", f)
	case err := <-errch:
		t.Errorf("unexpected purge error %v", err)
	case <-time.After(10 * time.Millisecond):
	}
	close(stop)
}

func TestPurgeFileHoldingLockFile(t *testing.T) {
	dir, err := ioutil.TempDir("", "purgefile")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	for i := 0; i < 10; i++ {
		var f *os.File
		f, err = os.Create(path.Join(dir, fmt.Sprintf("%d.test", i)))
		if err != nil {
			t.Fatal(err)
		}
		f.Close()
	}

	// create a purge barrier at 5
	p := path.Join(dir, fmt.Sprintf("%d.test", 5))
	l, err := LockFile(p, os.O_WRONLY, PrivateFileMode)
	if err != nil {
		t.Fatal(err)
	}

	stop, purgec := make(chan struct{}), make(chan string, 10)
	errch := purgeFile(dir, "test", 3, time.Millisecond, stop, purgec)

	for i := 0; i < 5; i++ {
		select {
		case <-purgec:
		case <-time.After(time.Second):
			t.Fatalf("purge took too long")
		}
	}

	fnames, rerr := ReadDir(dir)
	if rerr != nil {
		t.Fatal(rerr)
	}

	wnames := []string{"5.test", "6.test", "7.test", "8.test", "9.test"}
	if !reflect.DeepEqual(fnames, wnames) {
		t.Errorf("filenames = %v, want %v", fnames, wnames)
	}

	select {
	case s := <-purgec:
		t.Errorf("unexpected purge %q", s)
	case err = <-errch:
		t.Errorf("unexpected purge error %v", err)
	case <-time.After(10 * time.Millisecond):
	}

	// remove the purge barrier
	if err = l.Close(); err != nil {
		t.Fatal(err)
	}

	// wait for rest of purges (5, 6)
	for i := 0; i < 2; i++ {
		select {
		case <-purgec:
		case <-time.After(time.Second):
			t.Fatalf("purge took too long")
		}
	}

	fnames, rerr = ReadDir(dir)
	if rerr != nil {
		t.Fatal(rerr)
	}
	wnames = []string{"7.test", "8.test", "9.test"}
	if !reflect.DeepEqual(fnames, wnames) {
		t.Errorf("filenames = %v, want %v", fnames, wnames)
	}

	select {
	case f := <-purgec:
		t.Errorf("unexpected purge on %q", f)
	case err := <-errch:
		t.Errorf("unexpected purge error %v", err)
	case <-time.After(10 * time.Millisecond):
	}

	close(stop)
}
