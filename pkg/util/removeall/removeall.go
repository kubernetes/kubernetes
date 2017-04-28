/*
Copyright 2017 The Kubernetes Authors.

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

package removeall

import (
	"fmt"
	"io"
	"os"
	"syscall"

	"k8s.io/kubernetes/pkg/util/mount"
)

// RemoveAllOneFilesystem removes path and any children it contains.
// It removes everything it can but returns the first error
// it encounters. If the path does not exist, RemoveAll
// returns nil (no error).
// It makes sure it does not cross mount boundary, i.e. it does *not* remove
// files from another filesystems. Like 'rm -rf --one-file-system'.
// It is copied from RemoveAll() sources, with IsLikelyNotMountPoint
func RemoveAllOneFilesystem(mounter mount.Interface, path string) error {
	// Simple case: if Remove works, we're done.
	err := os.Remove(path)
	if err == nil || os.IsNotExist(err) {
		return nil
	}

	// Otherwise, is this a directory we need to recurse into?
	dir, serr := os.Lstat(path)
	if serr != nil {
		if serr, ok := serr.(*os.PathError); ok && (os.IsNotExist(serr.Err) || serr.Err == syscall.ENOTDIR) {
			return nil
		}
		return serr
	}
	if !dir.IsDir() {
		// Not a directory; return the error from Remove.
		return err
	}

	// Directory.
	isNotMount, err := mounter.IsLikelyNotMountPoint(path)
	if err != nil {
		return err
	}
	if !isNotMount {
		return fmt.Errorf("cannot delete directory %s: it is a mount point", path)
	}

	fd, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			// Race. It was deleted between the Lstat and Open.
			// Return nil per RemoveAll's docs.
			return nil
		}
		return err
	}

	// Remove contents & return first error.
	err = nil
	for {
		names, err1 := fd.Readdirnames(100)
		for _, name := range names {
			err1 := RemoveAllOneFilesystem(mounter, path+string(os.PathSeparator)+name)
			if err == nil {
				err = err1
			}
		}
		if err1 == io.EOF {
			break
		}
		// If Readdirnames returned an error, use it.
		if err == nil {
			err = err1
		}
		if len(names) == 0 {
			break
		}
	}

	// Close directory, because windows won't remove opened directory.
	fd.Close()

	// Remove directory.
	err1 := os.Remove(path)
	if err1 == nil || os.IsNotExist(err1) {
		return nil
	}
	if err == nil {
		err = err1
	}
	return err
}
