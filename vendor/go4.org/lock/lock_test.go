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
	"bufio"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
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

	var lk io.Closer
	for scan := bufio.NewScanner(os.Stdin); scan.Scan(); {
		var err error
		switch scan.Text() {
		case "lock":
			lk, err = lock(f)
		case "unlock":
			err = lk.Close()
			lk = nil
		case "exit":
			// Simulate a crash, or at least not unlocking the lock.
			os.Exit(0)
		default:
			err = fmt.Errorf("unexpected child command %q", scan.Text())
		}
		if err != nil {
			fmt.Println(err)
		} else {
			fmt.Println("")
		}
	}
}

func testLock(t *testing.T, portable bool) {
	lock := Lock
	if portable {
		lock = lockPortable
	}
	t.Logf("test lock, portable %v", portable)

	td, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(td)

	path := filepath.Join(td, "foo.lock")

	proc := newChildProc(t, path, portable)
	defer proc.kill()

	t.Logf("First lock in child")
	if err := proc.do("lock"); err != nil {
		t.Fatalf("first lock in child process: %v", err)
	}

	t.Logf("Crash child")
	if err := proc.do("exit"); err != nil {
		t.Fatalf("crash in child process: %v", err)
	}

	proc = newChildProc(t, path, portable)
	defer proc.kill()

	t.Logf("Locking+unlocking in child...")
	if err := proc.do("lock"); err != nil {
		t.Fatalf("lock in child process after crashing child: %v", err)
	}
	if err := proc.do("unlock"); err != nil {
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
	if err := proc.do("lock"); err == nil {
		t.Fatalf("expected lock in child process to fail")
	}

	t.Logf("Unlocking lock in parent")
	if err := lk1.Close(); err != nil {
		t.Fatal(err)
	}

	t.Logf("Trying lock again in child...")
	if err := proc.do("lock"); err != nil {
		t.Fatal(err)
	}
	if err := proc.do("unlock"); err != nil {
		t.Fatal(err)
	}

	lk3, err := lock(path)
	if err != nil {
		t.Fatal(err)
	}
	lk3.Close()
}

type childLockCmd struct {
	op    string
	reply chan<- error
}

type childProc struct {
	proc *os.Process
	c    chan childLockCmd
}

func (c *childProc) kill() {
	c.proc.Kill()
}

func (c *childProc) do(op string) error {
	reply := make(chan error)
	c.c <- childLockCmd{
		op:    op,
		reply: reply,
	}
	return <-reply
}

func newChildProc(t *testing.T, path string, portable bool) *childProc {
	cmd := exec.Command(os.Args[0], "-test.run=LockInChild$")
	cmd.Env = []string{"TEST_LOCK_FILE=" + path}
	toChild, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("cannot make pipe: %v", err)
	}
	fromChild, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("cannot make pipe: %v", err)
	}
	cmd.Stderr = os.Stderr
	if portable {
		cmd.Env = append(cmd.Env, "TEST_LOCK_PORTABLE=1")
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("cannot start child: %v", err)
	}
	cmdChan := make(chan childLockCmd)
	go func() {
		defer fromChild.Close()
		defer toChild.Close()
		inScan := bufio.NewScanner(fromChild)
		for c := range cmdChan {
			fmt.Fprintln(toChild, c.op)
			ok := inScan.Scan()
			if c.op == "exit" {
				if ok {
					c.reply <- errors.New("child did not exit")
				} else {
					cmd.Wait()
					c.reply <- nil
				}
				break
			}
			if !ok {
				panic("child exited early")
			}
			if errText := inScan.Text(); errText != "" {
				c.reply <- errors.New(errText)
			} else {
				c.reply <- nil
			}
		}
	}()
	return &childProc{
		c:    cmdChan,
		proc: cmd.Process,
	}
}
