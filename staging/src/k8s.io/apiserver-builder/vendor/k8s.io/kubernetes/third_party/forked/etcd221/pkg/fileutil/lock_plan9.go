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
	"errors"
	"os"
	"syscall"
	"time"
)

var (
	ErrLocked = errors.New("file already locked")
)

type Lock interface {
	Name() string
	TryLock() error
	Lock() error
	Unlock() error
	Destroy() error
}

type lock struct {
	fname string
	file  *os.File
}

func (l *lock) Name() string {
	return l.fname
}

// TryLock acquires exclusivity on the lock without blocking
func (l *lock) TryLock() error {
	err := os.Chmod(l.fname, syscall.DMEXCL|0600)
	if err != nil {
		return err
	}

	f, err := os.Open(l.fname)
	if err != nil {
		return ErrLocked
	}

	l.file = f
	return nil
}

// Lock acquires exclusivity on the lock with blocking
func (l *lock) Lock() error {
	err := os.Chmod(l.fname, syscall.DMEXCL|0600)
	if err != nil {
		return err
	}

	for {
		f, err := os.Open(l.fname)
		if err == nil {
			l.file = f
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// Unlock unlocks the lock
func (l *lock) Unlock() error {
	return l.file.Close()
}

func (l *lock) Destroy() error {
	return nil
}

func NewLock(file string) (Lock, error) {
	l := &lock{fname: file}
	return l, nil
}
