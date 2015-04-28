/*
Copyright 2013 The Go Authors

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

package lock

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
)

func TestLock(t *testing.T) {
	testLock(t, false)
}

func TestLockPortable(t *testing.T) {
	testLock(t, true)
}

func TestLockInChild(t *testing.T) {
	f := os.Getenv("TEST_LOCK_FILE")
	if f == "" {
		// not child
		return
	}
	lock := Lock
	if v, _ := strconv.ParseBool(os.Getenv("TEST_LOCK_PORTABLE")); v {
		lock = lockPortable
	}

	lk, err := lock(f)
	if err != nil {
		log.Fatalf("Lock failed: %v", err)
	}

	if v, _ := strconv.ParseBool(os.Getenv("TEST_LOCK_CRASH")); v {
		// Simulate a crash, or at least not unlocking the
		// lock.  We still exit 0 just to simplify the parent
		// process exec code.
		os.Exit(0)
	}
	lk.Close()
}

func testLock(t *testing.T, portable bool) {
	lock := Lock
	if portable {
		lock = lockPortable
	}

	td, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(td)

	path := filepath.Join(td, "foo.lock")

	childLock := func(crash bool) error {
		cmd := exec.Command(os.Args[0], "-test.run=LockInChild$")
		cmd.Env = []string{"TEST_LOCK_FILE=" + path}
		if portable {
			cmd.Env = append(cmd.Env, "TEST_LOCK_PORTABLE=1")
		}
		if crash {
			cmd.Env = append(cmd.Env, "TEST_LOCK_CRASH=1")
		}
		out, err := cmd.CombinedOutput()
		t.Logf("Child output: %q (err %v)", out, err)
		if err != nil {
			return fmt.Errorf("Child Process lock of %s failed: %v %s", path, err, out)
		}
		return nil
	}

	t.Logf("Locking in crashing child...")
	if err := childLock(true); err != nil {
		t.Fatalf("first lock in child process: %v", err)
	}

	t.Logf("Locking+unlocking in child...")
	if err := childLock(false); err != nil {
		t.Fatalf("lock in child process after crashing child: %v", err)
	}

	t.Logf("Locking in parent...")
	lk1, err := lock(path)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Again in parent...")
	_, err = lock(path)
	if err == nil {
		t.Fatal("expected second lock to fail")
	}

	t.Logf("Locking in child...")
	if childLock(false) == nil {
		t.Fatalf("expected lock in child process to fail")
	}

	t.Logf("Unlocking lock in parent")
	if err := lk1.Close(); err != nil {
		t.Fatal(err)
	}

	lk3, err := lock(path)
	if err != nil {
		t.Fatal(err)
	}
	lk3.Close()
}
