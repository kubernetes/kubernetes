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
	"io/ioutil"
	"os"
	"testing"
	"time"
)

func TestLockAndUnlock(t *testing.T) {
	f, err := ioutil.TempFile("", "lock")
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	defer func() {
		err = os.Remove(f.Name())
		if err != nil {
			t.Fatal(err)
		}
	}()

	// lock the file
	l, err := NewLock(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer l.Destroy()
	err = l.Lock()
	if err != nil {
		t.Fatal(err)
	}

	// try lock a locked file
	dupl, err := NewLock(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	err = dupl.TryLock()
	if err != ErrLocked {
		t.Errorf("err = %v, want %v", err, ErrLocked)
	}

	// unlock the file
	err = l.Unlock()
	if err != nil {
		t.Fatal(err)
	}

	// try lock the unlocked file
	err = dupl.TryLock()
	if err != nil {
		t.Errorf("err = %v, want %v", err, nil)
	}
	defer dupl.Destroy()

	// blocking on locked file
	locked := make(chan struct{}, 1)
	go func() {
		l.Lock()
		locked <- struct{}{}
	}()

	select {
	case <-locked:
		t.Error("unexpected unblocking")
	case <-time.After(100 * time.Millisecond):
	}

	// unlock
	err = dupl.Unlock()
	if err != nil {
		t.Fatal(err)
	}

	// the previously blocked routine should be unblocked
	select {
	case <-locked:
	case <-time.After(1 * time.Second):
		t.Error("unexpected blocking")
	}
}
