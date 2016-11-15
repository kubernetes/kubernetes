// +build linux darwin freebsd openbsd netbsd dragonfly

/*
Copyright 2016 The Kubernetes Authors.

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

package flock

import (
	"os"
	"sync"

	"golang.org/x/sys/unix"
)

var (
	// lock guards lockfile. Assignment is not atomic.
	lock sync.Mutex
	// os.File has a runtime.Finalizer so the fd will be closed if the struct
	// is garbage collected. Let's hold onto a reference so that doesn't happen.
	lockfile *os.File
)

// Acquire acquires a lock on a file for the duration of the process. This method
// is reentrant.
func Acquire(path string) error {
	lock.Lock()
	defer lock.Unlock()
	var err error
	if lockfile, err = os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0600); err != nil {
		return err
	}
	defer lockfile.Close()
	opts := unix.Flock_t{Type: unix.F_WRLCK}
	if err := unix.FcntlFlock(lockfile.Fd(), unix.F_SETLKW, &opts); err != nil {
		return err
	}
	return nil
}
