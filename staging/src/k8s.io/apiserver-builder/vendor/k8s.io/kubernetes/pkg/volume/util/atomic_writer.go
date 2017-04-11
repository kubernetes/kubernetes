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

package util

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/util/sets"
)

const (
	maxFileNameLength = 255
	maxPathLength     = 4096
)

// AtomicWriter handles atomically projecting content for a set of files into
// a target directory.
//
// Note:
//
// 1. AtomicWriter reserves the set of pathnames starting with `..`.
// 2. AtomicWriter offers no concurrency guarantees and must be synchronized
//    by the caller.
//
// The visible files in this volume are symlinks to files in the writer's data
// directory.  Actual files are stored in a hidden timestamped directory which
// is symlinked to by the data directory. The timestamped directory and
// data directory symlink are created in the writer's target dir.  This scheme
// allows the files to be atomically updated by changing the target of the
// data directory symlink.
//
// Consumers of the target directory can monitor the ..data symlink using
// inotify or fanotify to receive events when the content in the volume is
// updated.
type AtomicWriter struct {
	targetDir  string
	logContext string
}

type FileProjection struct {
	Data []byte
	Mode int32
}

// NewAtomicWriter creates a new AtomicWriter configured to write to the given
// target directory, or returns an error if the target directory does not exist.
func NewAtomicWriter(targetDir string, logContext string) (*AtomicWriter, error) {
	_, err := os.Stat(targetDir)
	if os.IsNotExist(err) {
		return nil, err
	}

	return &AtomicWriter{targetDir: targetDir, logContext: logContext}, nil
}

const (
	dataDirName    = "..data"
	newDataDirName = "..data_tmp"
)

// Write does an atomic projection of the given payload into the writer's target
// directory.  Input paths must not begin with '..'.
//
// The Write algorithm is:
//
//  1.  The payload is validated; if the payload is invalid, the function returns
//  2.  The user-visible portion of the volume is walked to determine whether any
//      portion of the payload was deleted and is still present on disk.
//      If the payload is already present on disk and there are no deleted files,
//      the function returns
//  3.  A check is made to determine whether data present in the payload has changed
//  4.  A new timestamped dir is created
//  5.  The payload is written to the new timestamped directory
//  6.  Symlinks and directory for new user-visible files are created (if needed).
//
//      For example, consider the files:
//        <target-dir>/podName
//        <target-dir>/user/labels
//        <target-dir>/k8s/annotations
//
//      The user visible files are symbolic links into the internal data directory:
//        <target-dir>/podName         -> ..data/podName
//        <target-dir>/usr/labels      -> ../..data/usr/labels
//        <target-dir>/k8s/annotations -> ../..data/k8s/annotations
//
//      Relative links are created into the data directory for files in subdirectories.
//
//      The data directory itself is a link to a timestamped directory with
//      the real data:
//        <target-dir>/..data          -> ..2016_02_01_15_04_05.12345678/
//  7.  The current timestamped directory is detected by reading the data directory
//      symlink
//  8.  A symlink to the new timestamped directory ..data_tmp is created that will
//      become the new data directory
//  9.  The new data directory symlink is renamed to the data directory; rename is atomic
// 10.  Old paths are removed from the user-visible portion of the target directory
// 11.  The previous timestamped directory is removed, if it exists
func (w *AtomicWriter) Write(payload map[string]FileProjection) error {
	// (1)
	cleanPayload, err := validatePayload(payload)
	if err != nil {
		glog.Errorf("%s: invalid payload: %v", w.logContext, err)
		return err
	}

	// (2)
	pathsToRemove, err := w.pathsToRemove(cleanPayload)
	if err != nil {
		glog.Errorf("%s: error determining user-visible files to remove: %v", w.logContext, err)
		return err
	}

	// (3)
	if should, err := w.shouldWritePayload(cleanPayload); err != nil {
		glog.Errorf("%s: error determining whether payload should be written to disk: %v", w.logContext, err)
		return err
	} else if !should && len(pathsToRemove) == 0 {
		glog.V(4).Infof("%s: no update required for target directory %v", w.logContext, w.targetDir)
		return nil
	} else {
		glog.V(4).Infof("%s: write required for target directory %v", w.logContext, w.targetDir)
	}

	// (4)
	tsDir, err := w.newTimestampDir()
	if err != nil {
		glog.V(4).Infof("%s: error creating new ts data directory: %v", w.logContext, err)
		return err
	}

	// (5)
	if err = w.writePayloadToDir(cleanPayload, tsDir); err != nil {
		glog.Errorf("%s: error writing payload to ts data directory %s: %v", w.logContext, tsDir, err)
		return err
	} else {
		glog.V(4).Infof("%s: performed write of new data to ts data directory: %s", w.logContext, tsDir)
	}

	// (6)
	if err = w.createUserVisibleFiles(cleanPayload); err != nil {
		glog.Errorf("%s: error creating visible symlinks in %s: %v", w.logContext, w.targetDir, err)
		return err
	}

	// (7)
	_, tsDirName := filepath.Split(tsDir)
	dataDirPath := path.Join(w.targetDir, dataDirName)
	oldTsDir, err := os.Readlink(dataDirPath)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("%s: error reading link for data directory: %v", w.logContext, err)
		return err
	}

	// (8)
	newDataDirPath := path.Join(w.targetDir, newDataDirName)
	if err = os.Symlink(tsDirName, newDataDirPath); err != nil {
		os.RemoveAll(tsDir)
		glog.Errorf("%s: error creating symbolic link for atomic update: %v", w.logContext, err)
		return err
	}

	// (9)
	if runtime.GOOS == "windows" {
		os.Remove(dataDirPath)
		err = os.Symlink(tsDirName, dataDirPath)
		os.Remove(newDataDirPath)
	} else {
		err = os.Rename(newDataDirPath, dataDirPath)
	}
	if err != nil {
		os.Remove(newDataDirPath)
		os.RemoveAll(tsDir)
		glog.Errorf("%s: error renaming symbolic link for data directory %s: %v", w.logContext, newDataDirPath, err)
		return err
	}

	// (10)
	if err = w.removeUserVisiblePaths(pathsToRemove); err != nil {
		glog.Errorf("%s: error removing old visible symlinks: %v", w.logContext, err)
		return err
	}

	// (11)
	if len(oldTsDir) > 0 {
		if err = os.RemoveAll(path.Join(w.targetDir, oldTsDir)); err != nil {
			glog.Errorf("%s: error removing old data directory %s: %v", w.logContext, oldTsDir, err)
			return err
		}
	}

	return nil
}

// validatePayload returns an error if any path in the payload  returns a copy of the payload with the paths cleaned.
func validatePayload(payload map[string]FileProjection) (map[string]FileProjection, error) {
	cleanPayload := make(map[string]FileProjection)
	for k, content := range payload {
		if err := validatePath(k); err != nil {
			return nil, err
		}

		cleanPayload[path.Clean(k)] = content
	}

	return cleanPayload, nil
}

// validatePath validates a single path, returning an error if the path is
// invalid.  paths may not:
//
// 1. be absolute
// 2. contain '..' as an element
// 3. start with '..'
// 4. contain filenames larger than 255 characters
// 5. be longer than 4096 characters
func validatePath(targetPath string) error {
	// TODO: somehow unify this with the similar api validation,
	// validateVolumeSourcePath; the error semantics are just different enough
	// from this that it was time-prohibitive trying to find the right
	// refactoring to re-use.
	if targetPath == "" {
		return fmt.Errorf("invalid path: must not be empty: %q", targetPath)
	}
	if path.IsAbs(targetPath) {
		return fmt.Errorf("invalid path: must be relative path: %s", targetPath)
	}

	if len(targetPath) > maxPathLength {
		return fmt.Errorf("invalid path: must be less than %d characters", maxPathLength)
	}

	items := strings.Split(targetPath, string(os.PathSeparator))
	for _, item := range items {
		if item == ".." {
			return fmt.Errorf("invalid path: must not contain '..': %s", targetPath)
		}
		if len(item) > maxFileNameLength {
			return fmt.Errorf("invalid path: filenames must be less than %d characters", maxFileNameLength)
		}
	}
	if strings.HasPrefix(items[0], "..") && len(items[0]) > 2 {
		return fmt.Errorf("invalid path: must not start with '..': %s", targetPath)
	}

	return nil
}

// shouldWritePayload returns whether the payload should be written to disk.
func (w *AtomicWriter) shouldWritePayload(payload map[string]FileProjection) (bool, error) {
	for userVisiblePath, fileProjection := range payload {
		shouldWrite, err := w.shouldWriteFile(path.Join(w.targetDir, userVisiblePath), fileProjection.Data)
		if err != nil {
			return false, err
		}

		if shouldWrite {
			return true, nil
		}
	}

	return false, nil
}

// shouldWriteFile returns whether a new version of a file should be written to disk.
func (w *AtomicWriter) shouldWriteFile(path string, content []byte) (bool, error) {
	_, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return true, nil
	}

	contentOnFs, err := ioutil.ReadFile(path)
	if err != nil {
		return false, err
	}

	return (bytes.Compare(content, contentOnFs) != 0), nil
}

// pathsToRemove walks the user-visible portion of the target directory and
// determines which paths should be removed (if any) after the payload is
// written to the target directory.
func (w *AtomicWriter) pathsToRemove(payload map[string]FileProjection) (sets.String, error) {
	paths := sets.NewString()
	visitor := func(path string, info os.FileInfo, err error) error {
		if path == w.targetDir {
			return nil
		}

		relativePath := strings.TrimPrefix(path, w.targetDir)
		if runtime.GOOS == "windows" {
			relativePath = strings.TrimPrefix(relativePath, "\\")
		} else {
			relativePath = strings.TrimPrefix(relativePath, "/")
		}
		if strings.HasPrefix(relativePath, "..") {
			return nil
		}

		paths.Insert(relativePath)
		return nil
	}

	err := filepath.Walk(w.targetDir, visitor)
	if os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	glog.V(5).Infof("%s: current paths:   %+v", w.targetDir, paths.List())

	newPaths := sets.NewString()
	for file := range payload {
		// add all subpaths for the payload to the set of new paths
		// to avoid attempting to remove non-empty dirs
		for subPath := file; subPath != ""; {
			newPaths.Insert(subPath)
			subPath, _ = filepath.Split(subPath)
			subPath = strings.TrimSuffix(subPath, "/")
		}
	}
	glog.V(5).Infof("%s: new paths:       %+v", w.targetDir, newPaths.List())

	result := paths.Difference(newPaths)
	glog.V(5).Infof("%s: paths to remove: %+v", w.targetDir, result)

	return result, nil
}

// newTimestampDir creates a new timestamp directory
func (w *AtomicWriter) newTimestampDir() (string, error) {
	tsDir, err := ioutil.TempDir(w.targetDir, fmt.Sprintf("..%s.", time.Now().Format("1981_02_01_15_04_05")))
	if err != nil {
		glog.Errorf("%s: unable to create new temp directory: %v", w.logContext, err)
		return "", err
	}

	// 0755 permissions are needed to allow 'group' and 'other' to recurse the
	// directory tree.  do a chmod here to ensure that permissions are set correctly
	// regardless of the process' umask.
	err = os.Chmod(tsDir, 0755)
	if err != nil {
		glog.Errorf("%s: unable to set mode on new temp directory: %v", w.logContext, err)
		return "", err
	}

	return tsDir, nil
}

// writePayloadToDir writes the given payload to the given directory.  The
// directory must exist.
func (w *AtomicWriter) writePayloadToDir(payload map[string]FileProjection, dir string) error {
	for userVisiblePath, fileProjection := range payload {
		content := fileProjection.Data
		mode := os.FileMode(fileProjection.Mode)
		fullPath := path.Join(dir, userVisiblePath)
		baseDir, _ := filepath.Split(fullPath)

		err := os.MkdirAll(baseDir, os.ModePerm)
		if err != nil {
			glog.Errorf("%s: unable to create directory %s: %v", w.logContext, baseDir, err)
			return err
		}

		err = ioutil.WriteFile(fullPath, content, mode)
		if err != nil {
			glog.Errorf("%s: unable to write file %s with mode %v: %v", w.logContext, fullPath, mode, err)
			return err
		}
		// Chmod is needed because ioutil.WriteFile() ends up calling
		// open(2) to create the file, so the final mode used is "mode &
		// ~umask". But we want to make sure the specified mode is used
		// in the file no matter what the umask is.
		err = os.Chmod(fullPath, mode)
		if err != nil {
			glog.Errorf("%s: unable to write file %s with mode %v: %v", w.logContext, fullPath, mode, err)
		}
	}

	return nil
}

// createUserVisibleFiles creates the relative symlinks for all the
// files configured in the payload. If the directory in a file path does not
// exist, it is created.
//
// Viz:
// For files: "bar", "foo/bar", "baz/bar", "foo/baz/blah"
// the following symlinks and subdirectories are created:
// bar          -> ..data/bar
// foo/bar      -> ../..data/foo/bar
// baz/bar      -> ../..data/baz/bar
// foo/baz/blah -> ../../..data/foo/baz/blah
func (w *AtomicWriter) createUserVisibleFiles(payload map[string]FileProjection) error {
	for userVisiblePath := range payload {
		dir, _ := filepath.Split(userVisiblePath)
		subDirs := 0
		if len(dir) > 0 {
			// If dir is not empty, the projection path contains at least one
			// subdirectory (example: userVisiblePath := "foo/bar").
			// Since filepath.Split leaves a trailing path separator, in this
			// example, dir = "foo/".  In order to calculate the number of
			// subdirectories, we must subtract 1 from the number returned by split.
			subDirs = len(strings.Split(dir, "/")) - 1
			err := os.MkdirAll(path.Join(w.targetDir, dir), os.ModePerm)
			if err != nil {
				return err
			}
		}
		_, err := os.Readlink(path.Join(w.targetDir, userVisiblePath))
		if err != nil && os.IsNotExist(err) {
			// The link into the data directory for this path doesn't exist; create it,
			// respecting the number of subdirectories necessary to link
			// correctly back into the data directory.
			visibleFile := path.Join(w.targetDir, userVisiblePath)
			dataDirFile := path.Join(strings.Repeat("../", subDirs), dataDirName, userVisiblePath)

			err = os.Symlink(dataDirFile, visibleFile)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// removeUserVisiblePaths removes the set of paths from the user-visible
// portion of the writer's target directory.
func (w *AtomicWriter) removeUserVisiblePaths(paths sets.String) error {
	orderedPaths := paths.List()
	for ii := len(orderedPaths) - 1; ii >= 0; ii-- {
		if err := os.Remove(path.Join(w.targetDir, orderedPaths[ii])); err != nil {
			glog.Errorf("%s: error pruning old user-visible path %s: %v", w.logContext, orderedPaths[ii], err)
			return err
		}
	}

	return nil
}
