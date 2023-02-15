//go:build !windows
// +build !windows

/*
Copyright 2019 The Kubernetes Authors.

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

package mount

import (
	"errors"
	"io/fs"
	"os"
	"syscall"

	"k8s.io/klog/v2"
)

// IsCorruptedMnt return true if err is about corrupted mount point
func IsCorruptedMnt(err error) bool {
	if err == nil {
		return false
	}
	var underlyingError error
	switch pe := err.(type) {
	case nil:
		return false
	case *os.PathError:
		underlyingError = pe.Err
	case *os.LinkError:
		underlyingError = pe.Err
	case *os.SyscallError:
		underlyingError = pe.Err
	case syscall.Errno:
		underlyingError = err
	}

	return underlyingError == syscall.ENOTCONN || underlyingError == syscall.ESTALE || underlyingError == syscall.EIO || underlyingError == syscall.EACCES || underlyingError == syscall.EHOSTDOWN
}

// PathExists returns true if the specified path exists.
// TODO: clean this up to use pkg/util/file/FileExists
func PathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	} else if errors.Is(err, fs.ErrNotExist) {
		err = syscall.Access(path, syscall.F_OK)
		if err == nil {
			// The access syscall says the file exists, the stat syscall says it
			// doesn't. This was observed on CIFS when the path was removed at
			// the server somehow. POSIX calls this a stale file handle, let's fake
			// that error and treat the path as existing but corrupted.
			klog.Warningf("Potential stale file handle detected: %s", path)
			return true, syscall.ESTALE
		}
		return false, nil
	} else if IsCorruptedMnt(err) {
		return true, err
	}
	return false, err
}
