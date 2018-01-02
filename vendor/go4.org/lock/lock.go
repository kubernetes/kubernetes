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

// Package lock is a file locking library.
package lock // import "go4.org/lock"

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
	abs, err := filepath.Abs(name)
	if err != nil {
		return nil, err
	}
	lockmu.Lock()
	defer lockmu.Unlock()
	if locked[abs] {
		return nil, fmt.Errorf("file %q already locked", abs)
	}

	c, err := lockFn(abs)
	if err != nil {
		return nil, fmt.Errorf("cannot acquire lock: %v", err)
	}
	locked[abs] = true
	return c, nil
}

var lockFn = lockPortable

// lockPortable is a portable version not using fcntl. Doesn't handle crashes as gracefully,
// since it can leave stale lock files.
func lockPortable(name string) (io.Closer, error) {
	fi, err := os.Stat(name)
	if err == nil && fi.Size() > 0 {
		st := portableLockStatus(name)
		switch st {
		case statusLocked:
			return nil, fmt.Errorf("file %q already locked", name)
		case statusStale:
			os.Remove(name)
		case statusInvalid:
			return nil, fmt.Errorf("can't Lock file %q: has invalid contents", name)
		}
	}
	f, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_TRUNC|os.O_EXCL, 0666)
	if err != nil {
		return nil, fmt.Errorf("failed to create lock file %s %v", name, err)
	}
	if err := json.NewEncoder(f).Encode(&pidLockMeta{OwnerPID: os.Getpid()}); err != nil {
		return nil, fmt.Errorf("cannot write owner pid: %v", err)
	}
	return &unlocker{
		f:        f,
		abs:      name,
		portable: true,
	}, nil
}

type lockStatus int

const (
	statusInvalid lockStatus = iota
	statusLocked
	statusUnlocked
	statusStale
)

type pidLockMeta struct {
	OwnerPID int
}

func portableLockStatus(path string) lockStatus {
	f, err := os.Open(path)
	if err != nil {
		return statusUnlocked
	}
	defer f.Close()
	var meta pidLockMeta
	if json.NewDecoder(f).Decode(&meta) != nil {
		return statusInvalid
	}
	if meta.OwnerPID == 0 {
		return statusInvalid
	}
	p, err := os.FindProcess(meta.OwnerPID)
	if err != nil {
		// e.g. on Windows
		return statusStale
	}
	// On unix, os.FindProcess always is true, so we have to send
	// it a signal to see if it's alive.
	if signalZero != nil {
		if p.Signal(signalZero) != nil {
			return statusStale
		}
	}
	return statusLocked
}

var signalZero os.Signal // nil or set by lock_sigzero.go

var (
	lockmu sync.Mutex
	locked = map[string]bool{} // abs path -> true
)

type unlocker struct {
	portable bool
	f        *os.File
	abs      string
	// once guards the close method call.
	once sync.Once
	// err holds the error returned by Close.
	err error
}

func (u *unlocker) Close() error {
	u.once.Do(u.close)
	return u.err
}

func (u *unlocker) close() {
	lockmu.Lock()
	defer lockmu.Unlock()
	delete(locked, u.abs)

	if u.portable {
		// In the portable lock implementation, it's
		// important to close before removing because
		// Windows won't allow us to remove an open
		// file.
		if err := u.f.Close(); err != nil {
			u.err = err
		}
		if err := os.Remove(u.abs); err != nil {
			// Note that if both Close and Remove fail,
			// we care more about the latter than the former
			// so we'll return that error.
			u.err = err
		}
		return
	}
	// In other implementatioons, it's nice for us to clean up.
	// If we do do this, though, it needs to be before the
	// u.f.Close below.
	os.Remove(u.abs)
	u.err = u.f.Close()
}
