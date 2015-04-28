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
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
)

// Lock locks the given file, creating the file if necessary. If the
// file already exists, it must have zero size or an error is returned.
// The lock is an exclusive lock (a write lock), but locked files
// should neither be read from nor written to. Such files should have
// zero size and only exist to co-ordinate ownership across processes.
//
// A nil Closer is returned if an error occurred. Otherwise, close that
// Closer to release the lock.
//
// On Linux, FreeBSD and OSX, a lock has the same semantics as fcntl(2)'s
// advisory locks.  In particular, closing any other file descriptor for the
// same file will release the lock prematurely.
//
// Attempting to lock a file that is already locked by the current process
// has undefined behavior.
//
// On other operating systems, lock will fallback to using the presence and
// content of a file named name + '.lock' to implement locking behavior.
func Lock(name string) (io.Closer, error) {
	return lockFn(name)
}

var lockFn = lockPortable

// Portable version not using fcntl. Doesn't handle crashes as gracefully,
// since it can leave stale lock files.
// TODO: write pid of owner to lock file and on race see if pid is
// still alive?
func lockPortable(name string) (io.Closer, error) {
	absName, err := filepath.Abs(name)
	if err != nil {
		return nil, fmt.Errorf("can't Lock file %q: can't find abs path: %v", name, err)
	}
	fi, err := os.Stat(absName)
	if err == nil && fi.Size() > 0 {
		if isStaleLock(absName) {
			os.Remove(absName)
		} else {
			return nil, fmt.Errorf("can't Lock file %q: has non-zero size", name)
		}
	}
	f, err := os.OpenFile(absName, os.O_RDWR|os.O_CREATE|os.O_TRUNC|os.O_EXCL, 0666)
	if err != nil {
		return nil, fmt.Errorf("failed to create lock file %s %v", absName, err)
	}
	if err := json.NewEncoder(f).Encode(&pidLockMeta{OwnerPID: os.Getpid()}); err != nil {
		return nil, err
	}
	return &lockCloser{f: f, abs: absName}, nil
}

type pidLockMeta struct {
	OwnerPID int
}

func isStaleLock(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	var meta pidLockMeta
	if json.NewDecoder(f).Decode(&meta) != nil {
		return false
	}
	if meta.OwnerPID == 0 {
		return false
	}
	p, err := os.FindProcess(meta.OwnerPID)
	if err != nil {
		// e.g. on Windows
		return true
	}
	// On unix, os.FindProcess always is true, so we have to send
	// it a signal to see if it's alive.
	if signalZero != nil {
		if p.Signal(signalZero) != nil {
			return true
		}
	}
	return false
}

var signalZero os.Signal // nil or set by lock_sigzero.go

type lockCloser struct {
	f    *os.File
	abs  string
	once sync.Once
	err  error
}

func (lc *lockCloser) Close() error {
	lc.once.Do(lc.close)
	return lc.err
}

func (lc *lockCloser) close() {
	if err := lc.f.Close(); err != nil {
		lc.err = err
	}
	if err := os.Remove(lc.abs); err != nil {
		lc.err = err
	}
}

var (
	lockmu sync.Mutex
	locked = map[string]bool{} // abs path -> true
)

// unlocker is used by the darwin and linux implementations with fcntl
// advisory locks.
type unlocker struct {
	f   *os.File
	abs string
}

func (u *unlocker) Close() error {
	lockmu.Lock()
	// Remove is not necessary but it's nice for us to clean up.
	// If we do do this, though, it needs to be before the
	// u.f.Close below.
	os.Remove(u.abs)
	if err := u.f.Close(); err != nil {
		return err
	}
	delete(locked, u.abs)
	lockmu.Unlock()
	return nil
}
