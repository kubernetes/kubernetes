// Copyright 2015 The rkt Authors
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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"

	"github.com/hashicorp/errwrap"
)

const (
	defaultDirPerm     os.FileMode = 0660
	defaultFilePerm    os.FileMode = 0660
	defaultLockRetries             = 3
)

type keyLockMode uint

const (
	keyLockExclusive keyLockMode = 1 << iota
	keyLockShared
	keyLockNonBlocking
)

// KeyLock is a lock for a specific key. The lock file is created inside a
// directory using the key name.
// This is useful when multiple processes want to take a lock but cannot use
// FileLock as they don't have a well defined file on the filesystem.
// key value must be a valid file name (as the lock file is named after the key
// value).
type KeyLock struct {
	lockDir string
	key     string
	// The lock on the key
	keyLock *FileLock
}

// NewKeyLock returns a KeyLock for the specified key without acquisition.
// lockdir is the directory where the lock file will be created. If lockdir
// doesn't exists it will be created.
// key value must be a valid file name (as the lock file is named after the key
// value).
func NewKeyLock(lockDir string, key string) (*KeyLock, error) {
	err := os.MkdirAll(lockDir, defaultDirPerm)
	if err != nil {
		return nil, err
	}
	keyLockFile := filepath.Join(lockDir, key)
	// create the file if it doesn't exists
	f, err := os.OpenFile(keyLockFile, os.O_RDONLY|os.O_CREATE, defaultFilePerm)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error creating key lock file"), err)
	}
	f.Close()
	keyLock, err := NewLock(keyLockFile, RegFile)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error opening key lock file"), err)
	}
	return &KeyLock{lockDir: lockDir, key: key, keyLock: keyLock}, nil
}

// Close closes the key lock which implicitly unlocks it as well
func (l *KeyLock) Close() {
	l.keyLock.Close()
}

// TryExclusiveLock takes an exclusive lock on a key without blocking.
// This is idempotent when the KeyLock already represents an exclusive lock,
// and tries promote a shared lock to exclusive atomically.
// It will return ErrLocked if any lock is already held on the key.
func (l *KeyLock) TryExclusiveKeyLock() error {
	return l.lock(keyLockExclusive|keyLockNonBlocking, defaultLockRetries)
}

// TryExclusiveLock takes an exclusive lock on the key without blocking.
// lockDir is the directory where the lock file will be created.
// It will return ErrLocked if any lock is already held.
func TryExclusiveKeyLock(lockDir string, key string) (*KeyLock, error) {
	return createAndLock(lockDir, key, keyLockExclusive|keyLockNonBlocking)
}

// ExclusiveLock takes an exclusive lock on a key.
// This is idempotent when the KeyLock already represents an exclusive lock,
// and promotes a shared lock to exclusive atomically.
// It will block if an exclusive lock is already held on the key.
func (l *KeyLock) ExclusiveKeyLock() error {
	return l.lock(keyLockExclusive, defaultLockRetries)
}

// ExclusiveLock takes an exclusive lock on a key.
// lockDir is the directory where the lock file will be created.
// It will block if an exclusive lock is already held on the key.
func ExclusiveKeyLock(lockDir string, key string) (*KeyLock, error) {
	return createAndLock(lockDir, key, keyLockExclusive)
}

// TrySharedLock takes a co-operative (shared) lock on the key without blocking.
// This is idempotent when the KeyLock already represents a shared lock,
// and tries demote an exclusive lock to shared atomically.
// It will return ErrLocked if an exclusive lock already exists on the key.
func (l *KeyLock) TrySharedKeyLock() error {
	return l.lock(keyLockShared|keyLockNonBlocking, defaultLockRetries)
}

// TrySharedLock takes a co-operative (shared) lock on a key without blocking.
// lockDir is the directory where the lock file will be created.
// It will return ErrLocked if an exclusive lock already exists on the key.
func TrySharedKeyLock(lockDir string, key string) (*KeyLock, error) {
	return createAndLock(lockDir, key, keyLockShared|keyLockNonBlocking)
}

// SharedLock takes a co-operative (shared) lock on a key.
// This is idempotent when the KeyLock already represents a shared lock,
// and demotes an exclusive lock to shared atomically.
// It will block if an exclusive lock is already held on the key.
func (l *KeyLock) SharedKeyLock() error {
	return l.lock(keyLockShared, defaultLockRetries)
}

// SharedLock takes a co-operative (shared) lock on a key.
// lockDir is the directory where the lock file will be created.
// It will block if an exclusive lock is already held on the key.
func SharedKeyLock(lockDir string, key string) (*KeyLock, error) {
	return createAndLock(lockDir, key, keyLockShared)
}

func createAndLock(lockDir string, key string, mode keyLockMode) (*KeyLock, error) {
	keyLock, err := NewKeyLock(lockDir, key)
	if err != nil {
		return nil, err
	}
	err = keyLock.lock(mode, defaultLockRetries)
	if err != nil {
		keyLock.Close()
		return nil, err
	}
	return keyLock, nil
}

// lock is the base function to take a lock and handle changed lock files
// As there's the need to remove unused (see CleanKeyLocks) lock files without
// races, a changed file detection is needed.
//
// Without changed file detection this can happen:
//
// Process A takes exclusive lock on file01
// Process B waits for exclusive lock on file01.
// Process A deletes file01 and then releases the lock.
// Process B takes the lock on the removed file01 as it has the fd opened
// Process C comes, creates the file as it doesn't exists, and it also takes an exclusive lock.
// Now B and C thinks to own an exclusive lock.
//
// maxRetries can be passed, useful for testing.
func (l *KeyLock) lock(mode keyLockMode, maxRetries int) error {
	retries := 0
	for {
		var err error
		var isExclusive bool
		var isNonBlocking bool
		if mode&keyLockExclusive != 0 {
			isExclusive = true
		}
		if mode&keyLockNonBlocking != 0 {
			isNonBlocking = true
		}
		switch {
		case isExclusive && !isNonBlocking:
			err = l.keyLock.ExclusiveLock()
		case isExclusive && isNonBlocking:
			err = l.keyLock.TryExclusiveLock()
		case !isExclusive && !isNonBlocking:
			err = l.keyLock.SharedLock()
		case !isExclusive && isNonBlocking:
			err = l.keyLock.TrySharedLock()
		}
		if err != nil {
			return err
		}

		// Check that the file referenced by the lock fd is the same as
		// the current file on the filesystem
		var lockStat, curStat syscall.Stat_t
		lfd, err := l.keyLock.Fd()
		if err != nil {
			return err
		}
		err = syscall.Fstat(lfd, &lockStat)
		if err != nil {
			return err
		}
		keyLockFile := filepath.Join(l.lockDir, l.key)
		fd, err := syscall.Open(keyLockFile, syscall.O_RDONLY, 0)
		// If there's an error opening the file return an error
		if err != nil {
			return err
		}
		if err := syscall.Fstat(fd, &curStat); err != nil {
			syscall.Close(fd)
			return err
		}
		syscall.Close(fd)
		if lockStat.Ino == curStat.Ino && lockStat.Dev == curStat.Dev {
			return nil
		}
		if retries >= maxRetries {
			return fmt.Errorf("cannot acquire lock after %d retries", retries)
		}

		// If the file has changed discard this lock and try to take another lock.
		l.keyLock.Close()
		nl, err := NewKeyLock(l.lockDir, l.key)
		if err != nil {
			return err
		}
		l.keyLock = nl.keyLock

		retries++
	}
}

// Unlock unlocks the key lock.
func (l *KeyLock) Unlock() error {
	err := l.keyLock.Unlock()
	if err != nil {
		return err
	}
	return nil
}

// CleanKeyLocks remove lock files from the lockDir.
// For every key it tries to take an Exclusive lock on it and skip it if it
// fails with ErrLocked
func CleanKeyLocks(lockDir string) error {
	f, err := os.Open(lockDir)
	if err != nil {
		return errwrap.Wrap(errors.New("error opening lockDir"), err)
	}
	defer f.Close()
	files, err := f.Readdir(0)
	if err != nil {
		return errwrap.Wrap(errors.New("error getting lock files list"), err)
	}
	for _, f := range files {
		filename := filepath.Join(lockDir, f.Name())
		keyLock, err := TryExclusiveKeyLock(lockDir, f.Name())
		if err == ErrLocked {
			continue
		}
		if err != nil {
			return err
		}

		err = os.Remove(filename)
		if err != nil {
			keyLock.Close()
			return errwrap.Wrap(errors.New("error removing lock file"), err)
		}
		keyLock.Close()
	}
	return nil
}
