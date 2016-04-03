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

// Package lock implements simple locking primitives on a
// regular file or directory using flock
package lock

import (
	"errors"
	"syscall"
)

var (
	ErrLocked     = errors.New("file already locked")
	ErrNotExist   = errors.New("file does not exist")
	ErrPermission = errors.New("permission denied")
	ErrNotRegular = errors.New("not a regular file")
)

// FileLock represents a lock on a regular file or a directory
type FileLock struct {
	path string
	fd   int
}

type LockType int

const (
	Dir LockType = iota
	RegFile
)

// TryExclusiveLock takes an exclusive lock without blocking.
// This is idempotent when the Lock already represents an exclusive lock,
// and tries promote a shared lock to exclusive atomically.
// It will return ErrLocked if any lock is already held.
func (l *FileLock) TryExclusiveLock() error {
	err := syscall.Flock(l.fd, syscall.LOCK_EX|syscall.LOCK_NB)
	if err == syscall.EWOULDBLOCK {
		err = ErrLocked
	}
	return err
}

// TryExclusiveLock takes an exclusive lock on a file/directory without blocking.
// It will return ErrLocked if any lock is already held on the file/directory.
func TryExclusiveLock(path string, lockType LockType) (*FileLock, error) {
	l, err := NewLock(path, lockType)
	if err != nil {
		return nil, err
	}
	err = l.TryExclusiveLock()
	if err != nil {
		return nil, err
	}
	return l, err
}

// ExclusiveLock takes an exclusive lock.
// This is idempotent when the Lock already represents an exclusive lock,
// and promotes a shared lock to exclusive atomically.
// It will block if an exclusive lock is already held.
func (l *FileLock) ExclusiveLock() error {
	return syscall.Flock(l.fd, syscall.LOCK_EX)
}

// ExclusiveLock takes an exclusive lock on a file/directory.
// It will block if an exclusive lock is already held on the file/directory.
func ExclusiveLock(path string, lockType LockType) (*FileLock, error) {
	l, err := NewLock(path, lockType)
	if err == nil {
		err = l.ExclusiveLock()
	}
	if err != nil {
		return nil, err
	}
	return l, nil
}

// TrySharedLock takes a co-operative (shared) lock without blocking.
// This is idempotent when the Lock already represents a shared lock,
// and tries demote an exclusive lock to shared atomically.
// It will return ErrLocked if an exclusive lock already exists.
func (l *FileLock) TrySharedLock() error {
	err := syscall.Flock(l.fd, syscall.LOCK_SH|syscall.LOCK_NB)
	if err == syscall.EWOULDBLOCK {
		err = ErrLocked
	}
	return err
}

// TrySharedLock takes a co-operative (shared) lock on a file/directory without blocking.
// It will return ErrLocked if an exclusive lock already exists on the file/directory.
func TrySharedLock(path string, lockType LockType) (*FileLock, error) {
	l, err := NewLock(path, lockType)
	if err != nil {
		return nil, err
	}
	err = l.TrySharedLock()
	if err != nil {
		return nil, err
	}
	return l, nil
}

// SharedLock takes a co-operative (shared) lock on.
// This is idempotent when the Lock already represents a shared lock,
// and demotes an exclusive lock to shared atomically.
// It will block if an exclusive lock is already held.
func (l *FileLock) SharedLock() error {
	return syscall.Flock(l.fd, syscall.LOCK_SH)
}

// SharedLock takes a co-operative (shared) lock on a file/directory.
// It will block if an exclusive lock is already held on the file/directory.
func SharedLock(path string, lockType LockType) (*FileLock, error) {
	l, err := NewLock(path, lockType)
	if err != nil {
		return nil, err
	}
	err = l.SharedLock()
	if err != nil {
		return nil, err
	}
	return l, nil
}

// Unlock unlocks the lock
func (l *FileLock) Unlock() error {
	return syscall.Flock(l.fd, syscall.LOCK_UN)
}

// Fd returns the lock's file descriptor, or an error if the lock is closed
func (l *FileLock) Fd() (int, error) {
	var err error
	if l.fd == -1 {
		err = errors.New("lock closed")
	}
	return l.fd, err
}

// Close closes the lock which implicitly unlocks it as well
func (l *FileLock) Close() error {
	fd := l.fd
	l.fd = -1
	return syscall.Close(fd)
}

// NewLock opens a new lock on a file without acquisition
func NewLock(path string, lockType LockType) (*FileLock, error) {
	l := &FileLock{path: path, fd: -1}

	mode := syscall.O_RDONLY | syscall.O_CLOEXEC
	if lockType == Dir {
		mode |= syscall.O_DIRECTORY
	}
	lfd, err := syscall.Open(l.path, mode, 0)
	if err != nil {
		if err == syscall.ENOENT {
			err = ErrNotExist
		} else if err == syscall.EACCES {
			err = ErrPermission
		}
		return nil, err
	}
	l.fd = lfd

	var stat syscall.Stat_t
	err = syscall.Fstat(lfd, &stat)
	if err != nil {
		return nil, err
	}
	// Check if the file is a regular file
	if lockType == RegFile && !(stat.Mode&syscall.S_IFMT == syscall.S_IFREG) {
		return nil, ErrNotRegular
	}

	return l, nil
}
