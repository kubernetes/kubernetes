// Copyright 2015 CoreOS, Inc.
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

	for i := 0; i < 5; i++ {
		_, err = os.Create(path.Join(dir, fmt.Sprintf("%d.test", i)))
		if err != nil {
			t.Fatal(err)
		}
	}

	stop := make(chan struct{})
	errch := PurgeFile(dir, "test", 3, time.Millisecond, stop)
	for i := 5; i < 10; i++ {
		_, err = os.Create(path.Join(dir, fmt.Sprintf("%d.test", i)))
		if err != nil {
			t.Fatal(err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	fnames, err := ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	wnames := []string{"7.test", "8.test", "9.test"}
	if !reflect.DeepEqual(fnames, wnames) {
		t.Errorf("filenames = %v, want %v", fnames, wnames)
	}
	select {
	case err := <-errch:
		t.Errorf("unexpected purge error %v", err)
	case <-time.After(time.Millisecond):
	}
	close(stop)
}

func TestPurgeFileHoldingLock(t *testing.T) {
	dir, err := ioutil.TempDir("", "purgefile")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	for i := 0; i < 10; i++ {
		_, err = os.Create(path.Join(dir, fmt.Sprintf("%d.test", i)))
		if err != nil {
			t.Fatal(err)
		}
	}

	// create a purge barrier at 5
	l, err := NewLock(path.Join(dir, fmt.Sprintf("%d.test", 5)))
	err = l.Lock()
	if err != nil {
		t.Fatal(err)
	}

	stop := make(chan struct{})
	errch := PurgeFile(dir, "test", 3, time.Millisecond, stop)
	time.Sleep(20 * time.Millisecond)

	fnames, err := ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	wnames := []string{"5.test", "6.test", "7.test", "8.test", "9.test"}
	if !reflect.DeepEqual(fnames, wnames) {
		t.Errorf("filenames = %v, want %v", fnames, wnames)
	}
	select {
	case err := <-errch:
		t.Errorf("unexpected purge error %v", err)
	case <-time.After(time.Millisecond):
	}

	// remove the purge barrier
	err = l.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	err = l.Destroy()
	if err != nil {
		t.Fatal(err)
	}

	time.Sleep(20 * time.Millisecond)

	fnames, err = ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	wnames = []string{"7.test", "8.test", "9.test"}
	if !reflect.DeepEqual(fnames, wnames) {
		t.Errorf("filenames = %v, want %v", fnames, wnames)
	}
	select {
	case err := <-errch:
		t.Errorf("unexpected purge error %v", err)
	case <-time.After(time.Millisecond):
	}

	close(stop)
}
