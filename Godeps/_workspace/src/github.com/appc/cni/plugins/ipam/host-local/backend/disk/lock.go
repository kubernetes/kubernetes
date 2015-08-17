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

package disk

import (
	"os"
	"syscall"
)

// FileLock wraps os.File to be used as a lock using flock
type FileLock struct {
	f *os.File
}

// NewFileLock opens file/dir at path and returns unlocked FileLock object
func NewFileLock(path string) (*FileLock, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	return &FileLock{f}, nil
}

// Close closes underlying file
func (l *FileLock) Close() error {
	return l.f.Close()
}

// Lock acquires an exclusive lock
func (l *FileLock) Lock() error {
	return syscall.Flock(int(l.f.Fd()), syscall.LOCK_EX)
}

// Unlock releases the lock
func (l *FileLock) Unlock() error {
	return syscall.Flock(int(l.f.Fd()), syscall.LOCK_UN)
}
