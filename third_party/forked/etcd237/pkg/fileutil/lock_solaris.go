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

// +build solaris

package fileutil

import (
	"errors"
	"os"
	"syscall"
)

var (
	ErrLocked = errors.New("file already locked")
)

type lock struct {
	fd   int
	file *os.File
}

func (l *lock) Name() string {
	return l.file.Name()
}

func (l *lock) TryLock() error {
	var lock syscall.Flock_t
	lock.Start = 0
	lock.Len = 0
	lock.Pid = 0
	lock.Type = syscall.F_WRLCK
	lock.Whence = 0
	lock.Pid = 0
	err := syscall.FcntlFlock(uintptr(l.fd), syscall.F_SETLK, &lock)
	if err != nil && err == syscall.EAGAIN {
		return ErrLocked
	}
	return err
}

func (l *lock) Lock() error {
	var lock syscall.Flock_t
	lock.Start = 0
	lock.Len = 0
	lock.Type = syscall.F_WRLCK
	lock.Whence = 0
	lock.Pid = 0
	return syscall.FcntlFlock(uintptr(l.fd), syscall.F_SETLK, &lock)
}

func (l *lock) Unlock() error {
	var lock syscall.Flock_t
	lock.Start = 0
	lock.Len = 0
	lock.Type = syscall.F_UNLCK
	lock.Whence = 0
	err := syscall.FcntlFlock(uintptr(l.fd), syscall.F_SETLK, &lock)
	if err != nil && err == syscall.EAGAIN {
		return ErrLocked
	}
	return err
}

func (l *lock) Destroy() error {
	return l.file.Close()
}

func NewLock(file string) (Lock, error) {
	f, err := os.OpenFile(file, os.O_WRONLY, 0600)
	if err != nil {
		return nil, err
	}
	l := &lock{int(f.Fd()), f}
	return l, nil
}
