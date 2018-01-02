// Copyright 2014 The rkt Authors
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

package lock

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestNewLock(t *testing.T) {
	f, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatalf("error creating tmpfile: %v", err)
	}
	defer os.Remove(f.Name())
	f.Close()

	l, err := NewLock(f.Name(), RegFile)
	if err != nil {
		t.Fatalf("error creating NewFileLock: %v", err)
	}
	l.Close()

	d, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("error creating tmpdir: %v", err)
	}
	defer os.Remove(d)

	l, err = NewLock(d, Dir)
	if err != nil {
		t.Fatalf("error creating NewLock: %v", err)
	}

	err = l.Close()
	if err != nil {
		t.Fatalf("error unlocking lock: %v", err)
	}

	if err = os.Remove(d); err != nil {
		t.Fatalf("error removing tmpdir: %v", err)
	}

	l, err = NewLock(d, Dir)
	if err == nil {
		t.Fatalf("expected error creating lock on nonexistent path")
	}
}

func TestExclusiveLock(t *testing.T) {
	dir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("error creating tmpdir: %v", err)
	}
	defer os.Remove(dir)

	// Set up the initial exclusive lock
	l, err := ExclusiveLock(dir, Dir)
	if err != nil {
		t.Fatalf("error creating lock: %v", err)
	}

	// reacquire the exclusive lock using the receiver interface
	err = l.TryExclusiveLock()
	if err != nil {
		t.Fatalf("error reacquiring exclusive lock: %v", err)
	}

	// Now try another exclusive lock, should fail
	_, err = TryExclusiveLock(dir, Dir)
	if err == nil {
		t.Fatalf("expected err trying exclusive lock")
	}

	// Unlock the original lock
	err = l.Close()
	if err != nil {
		t.Fatalf("error closing lock: %v", err)
	}

	// Now another exclusive lock should succeed
	_, err = TryExclusiveLock(dir, Dir)
	if err != nil {
		t.Fatalf("error creating lock: %v", err)
	}
}

func TestSharedLock(t *testing.T) {
	dir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("error creating tmpdir: %v", err)
	}
	defer os.Remove(dir)

	// Set up the initial shared lock
	l1, err := SharedLock(dir, Dir)
	if err != nil {
		t.Fatalf("error creating new shared lock: %v", err)
	}

	err = l1.TrySharedLock()
	if err != nil {
		t.Fatalf("error reacquiring shared lock: %v", err)
	}

	// Subsequent shared locks should succeed
	l2, err := TrySharedLock(dir, Dir)
	if err != nil {
		t.Fatalf("error creating shared lock: %v", err)
	}
	l3, err := TrySharedLock(dir, Dir)
	if err != nil {
		t.Fatalf("error creating shared lock: %v", err)
	}

	// But an exclusive lock should fail
	_, err = TryExclusiveLock(dir, Dir)
	if err == nil {
		t.Fatal("expected exclusive lock to fail")
	}

	// Close the locks
	err = l1.Close()
	if err != nil {
		t.Fatalf("error closing lock: %v", err)
	}
	err = l2.Close()
	if err != nil {
		t.Fatalf("error closing lock: %v", err)
	}

	// Only unlock one of them
	err = l3.Unlock()
	if err != nil {
		t.Fatalf("error unlocking lock: %v", err)
	}

	// Now try an exclusive lock, should succeed
	_, err = TryExclusiveLock(dir, Dir)
	if err != nil {
		t.Fatalf("error creating lock: %v", err)
	}
}
