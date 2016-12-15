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

// +build linux

package fileutil

import (
	"os"
	"syscall"
)

// Preallocate tries to allocate the space for given
// file. This operation is only supported on linux by a
// few filesystems (btrfs, ext4, etc.).
// If the operation is unsupported, no error will be returned.
// Otherwise, the error encountered will be returned.
func Preallocate(f *os.File, sizeInBytes int) error {
	// use mode = 1 to keep size
	// see FALLOC_FL_KEEP_SIZE
	err := syscall.Fallocate(int(f.Fd()), 1, 0, int64(sizeInBytes))
	if err != nil {
		errno, ok := err.(syscall.Errno)
		// treat not support as nil error
		if ok && errno == syscall.ENOTSUP {
			return nil
		}
		return err
	}
	return nil
}
