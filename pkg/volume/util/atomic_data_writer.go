/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/golang/glog"

	utilErrors "k8s.io/kubernetes/pkg/util/errors"
)

// AtomicDataWriter writes key/value pairs to files to the volume root
// specified using the given WriteFile function.
type AtomicDataWriter struct {
	VolumeRoot string

	WriteFile func(string, string) error
}

const (
	timeStampFormat = "2006_01_02_15_04_05"

	// It seems reasonable to allow dot-files in the config,
	// so we reserved double-dot-files for the implementation.
	atomicWriteDir    = "..atomic_write"
	atomicWriteTmpDir = "..atomic_write_tmp"
)

func timeStamp() string {
	return time.Now().Format(timeStampFormat)
}

// newTimeStampedDir writes the latest files into a new temporary directory with a timestamp.
func (w *AtomicDataWriter) newTimeStampedDir(files map[string]string) (string, error) {
	errlist := []error{}
	timestampDir, err := ioutil.TempDir(w.VolumeRoot, ".."+timeStamp())
	if err != nil {
		glog.Errorf("Unable to create a temporary directory: %s", err)
		return "", err
	}

	for fname, values := range files {
		fullPathFile := path.Join(timestampDir, fname)
		dir := filepath.Dir(fullPathFile)

		if err := os.MkdirAll(dir, os.ModePerm); err != nil {
			glog.Errorf("Unable to create directory `%s`: %s", dir, err)
			return "", err
		}

		if err := w.WriteFile(fullPathFile, values); err != nil {
			glog.Errorf("Unable to write file `%s`: %s", fullPathFile, err)
			errlist = append(errlist, err)
		}
	}
	return timestampDir, utilErrors.NewAggregate(errlist)
}

// updateSymlinksToCurrentDir creates the relative symlinks for all the files configured in this volume.
// If the directory in a file path does not exist, it is created.
//
// For example for files: "bar", "foo/bar", "baz/bar", "foo/baz/blah"
// the following symlinks and subdirectory are created:
// bar          -> ..atomic_write/bar
// baz/bar      -> ../..atomic_write/baz/bar
// foo/bar      -> ../..atomic_write/foo/bar
// foo/baz/blah -> ../../..atomic_write/foo/baz/blah
func (w *AtomicDataWriter) updateSymlinksToCurrentDir(files map[string]string) error {
	for fname := range files {
		dir := filepath.Dir(fname)

		subdirCount := 0
		if len(dir) > 0 {
			// if dir is not empty fname contains at least a subdirectory (for example: fname="foo/bar")
			// since filepath.Split leaves a trailing '/'  we have dir="foo/"
			// and since len(strings.Split"foo/")=2 to count the number
			// of sub directory you need to remove 1
			subdirCount = len(strings.Split(dir, "/")) - 1
			if err := os.MkdirAll(path.Join(w.VolumeRoot, dir), os.ModePerm); err != nil {
				return err
			}
		}

		if _, err := os.Readlink(path.Join(w.VolumeRoot, fname)); err != nil {
			// link does not exist create it
			presentedFile := path.Join(strings.Repeat("../", subdirCount), atomicWriteDir, fname)
			actualFile := path.Join(w.VolumeRoot, fname)
			if err := os.Symlink(presentedFile, actualFile); err != nil {
				return err
			}
		}
	}

	return nil
}

// Write writes requested key/value pairs in specified files.
//
// The file visible in this volume are symlinks to files in the '..atomic_write'
// directory. Actual files are stored in an hidden timestamped directory which is
// symlinked to by '..atomic_write'. The timestamped directory and '..atomic_write' symlink
// are created in the plugin root dir.  This scheme allows the files to be
// atomically updated by changing the target of the '..atomic_write' symlink.  When new
// data is available:
//
// 1.  A new timestamped dir is created and requested data is written inside
//     the new timestamped directory.
// 2.  Symlinks and directory for new files are created (if needed).
//     For example for files:
//       <volume-dir>/user_space/labels
//       <volume-dir>/k8s_space/annotations
//       <volume-dir>/podName
//     This structure is created:
//       <volume-dir>/podName               -> ..atomic_write/podName
//       <volume-dir>/user_space/labels     -> ../..atomic_write/user_space/labels
//       <volume-dir>/k8s_space/annotations -> ../..atomic_write/k8s_space/annotations
//       <volume-dir>/..atomic_write         -> ..atomic_write.12345678
//}     where ..atomic_write.12345678 is a randomly generated directory which contains
//     the real data. If a file has to be dumped in subdirectory (for example <volume-dir>/user_space/labels)
//     plugin builds a relative symlink (<volume-dir>/user_space/labels -> ../..atomic_write/user_space/labels)
// 3.  The previous timestamped directory is detected reading the '..atomic_write' symlink
// 4.  In case no symlink exists then it's created
// 5.  In case symlink exists a new temporary symlink is created ..atomic_write_tmp
// 6.  ..atomic_write_tmp is renamed to ..atomic_write
// 7.  The previous timestamped directory is removed
func (w *AtomicDataWriter) Write(files map[string]string) error {
	timestampDir, err := w.newTimeStampedDir(files)
	if err != nil {
		glog.Errorf("Unable to write files to the temporary directory: %s", err)
		return err
	}

	// update symbolic links for relative paths
	if err := w.updateSymlinksToCurrentDir(files); err != nil {
		os.RemoveAll(timestampDir)
		glog.Errorf("Unable to create symlinks and/or directory: %s", err)
		return err
	}

	oldTimestampDirectory, err := os.Readlink(path.Join(w.VolumeRoot, atomicWriteDir))
	if err != nil {
		glog.Errorf("Unable to read symbolic link: %s", err)

		return err
	}

	if err := os.Symlink(filepath.Base(timestampDir), path.Join(w.VolumeRoot, atomicWriteTmpDir)); err != nil {
		os.RemoveAll(timestampDir)
		glog.Errorf("Unable to create symolic link: %s", err)
		return err
	}

	// Rename the symbolic link atomicWriteTmpDir to atomicWriteDir
	if err := os.Rename(path.Join(w.VolumeRoot, atomicWriteTmpDir), path.Join(w.VolumeRoot, atomicWriteDir)); err != nil {
		// in case of error remove latest files and atomicWriteTmpDir
		os.Remove(path.Join(w.VolumeRoot, atomicWriteTmpDir))
		os.RemoveAll(timestampDir)
		glog.Errorf("Unable to rename symbolic link: %s", err)
		return err
	}

	// Remove oldTimestampDirectory
	if len(oldTimestampDirectory) > 0 {
		if err := os.RemoveAll(path.Join(w.VolumeRoot, oldTimestampDirectory)); err != nil {
			glog.Errorf("Unable to remove directory: %s", err)
			return err
		}
	}

	return nil
}
