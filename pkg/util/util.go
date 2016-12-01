/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"io"
	"os"
	"reflect"
	"regexp"
	"syscall"

	"k8s.io/kubernetes/pkg/util/mount"
)

// Takes a list of strings and compiles them into a list of regular expressions
func CompileRegexps(regexpStrings []string) ([]*regexp.Regexp, error) {
	regexps := []*regexp.Regexp{}
	for _, regexpStr := range regexpStrings {
		r, err := regexp.Compile(regexpStr)
		if err != nil {
			return []*regexp.Regexp{}, err
		}
		regexps = append(regexps, r)
	}
	return regexps, nil
}

// Detects if using systemd as the init system
// Please note that simply reading /proc/1/cmdline can be misleading because
// some installation of various init programs can automatically make /sbin/init
// a symlink or even a renamed version of their main program.
// TODO(dchen1107): realiably detects the init system using on the system:
// systemd, upstart, initd, etc.
func UsingSystemdInitSystem() bool {
	if _, err := os.Stat("/run/systemd/system"); err == nil {
		return true
	}

	return false
}

// Tests whether all pointer fields in a struct are nil.  This is useful when,
// for example, an API struct is handled by plugins which need to distinguish
// "no plugin accepted this spec" from "this spec is empty".
//
// This function is only valid for structs and pointers to structs.  Any other
// type will cause a panic.  Passing a typed nil pointer will return true.
func AllPtrFieldsNil(obj interface{}) bool {
	v := reflect.ValueOf(obj)
	if !v.IsValid() {
		panic(fmt.Sprintf("reflect.ValueOf() produced a non-valid Value for %#v", obj))
	}
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return true
		}
		v = v.Elem()
	}
	for i := 0; i < v.NumField(); i++ {
		if v.Field(i).Kind() == reflect.Ptr && !v.Field(i).IsNil() {
			return false
		}
	}
	return true
}

func FileExists(filename string) (bool, error) {
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return false, err
	}
	return true, nil
}

// ReadDirNoStat returns a string of files/directories contained
// in dirname without calling lstat on them.
func ReadDirNoStat(dirname string) ([]string, error) {
	if dirname == "" {
		dirname = "."
	}

	f, err := os.Open(dirname)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return f.Readdirnames(-1)
}

// IntPtr returns a pointer to an int
func IntPtr(i int) *int {
	o := i
	return &o
}

// Int32Ptr returns a pointer to an int32
func Int32Ptr(i int32) *int32 {
	o := i
	return &o
}

// IntPtrDerefOr dereference the int ptr and returns it i not nil,
// else returns def.
func IntPtrDerefOr(ptr *int, def int) int {
	if ptr != nil {
		return *ptr
	}
	return def
}

// Int32PtrDerefOr dereference the int32 ptr and returns it i not nil,
// else returns def.
func Int32PtrDerefOr(ptr *int32, def int32) int32 {
	if ptr != nil {
		return *ptr
	}
	return def
}

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
