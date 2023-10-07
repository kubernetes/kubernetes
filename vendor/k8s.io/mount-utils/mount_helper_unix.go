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
	"fmt"
	"io/fs"
	"os"
	"strconv"
	"strings"
	"syscall"

	"k8s.io/klog/v2"
	utilio "k8s.io/utils/io"
)

const (
	// At least number of fields per line in /proc/<pid>/mountinfo.
	expectedAtLeastNumFieldsPerMountInfo = 10
	// How many times to retry for a consistent read of /proc/mounts.
	maxListTries = 10
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

// MountInfo represents a single line in /proc/<pid>/mountinfo.
type MountInfo struct { // nolint: golint
	// Unique ID for the mount (maybe reused after umount).
	ID int
	// The ID of the parent mount (or of self for the root of this mount namespace's mount tree).
	ParentID int
	// Major indicates one half of the device ID which identifies the device class
	// (parsed from `st_dev` for files on this filesystem).
	Major int
	// Minor indicates one half of the device ID which identifies a specific
	// instance of device (parsed from `st_dev` for files on this filesystem).
	Minor int
	// The pathname of the directory in the filesystem which forms the root of this mount.
	Root string
	// Mount source, filesystem-specific information. e.g. device, tmpfs name.
	Source string
	// Mount point, the pathname of the mount point.
	MountPoint string
	// Optional fieds, zero or more fields of the form "tag[:value]".
	OptionalFields []string
	// The filesystem type in the form "type[.subtype]".
	FsType string
	// Per-mount options.
	MountOptions []string
	// Per-superblock options.
	SuperOptions []string
}

// ParseMountInfo parses /proc/xxx/mountinfo.
func ParseMountInfo(filename string) ([]MountInfo, error) {
	content, err := utilio.ConsistentRead(filename, maxListTries)
	if err != nil {
		return []MountInfo{}, err
	}
	contentStr := string(content)
	infos := []MountInfo{}

	for _, line := range strings.Split(contentStr, "\n") {
		if line == "" {
			// the last split() item is empty string following the last \n
			continue
		}
		// See `man proc` for authoritative description of format of the file.
		fields := strings.Fields(line)
		if len(fields) < expectedAtLeastNumFieldsPerMountInfo {
			return nil, fmt.Errorf("wrong number of fields in (expected at least %d, got %d): %s", expectedAtLeastNumFieldsPerMountInfo, len(fields), line)
		}
		id, err := strconv.Atoi(fields[0])
		if err != nil {
			return nil, err
		}
		parentID, err := strconv.Atoi(fields[1])
		if err != nil {
			return nil, err
		}
		mm := strings.Split(fields[2], ":")
		if len(mm) != 2 {
			return nil, fmt.Errorf("parsing '%s' failed: unexpected minor:major pair %s", line, mm)
		}
		major, err := strconv.Atoi(mm[0])
		if err != nil {
			return nil, fmt.Errorf("parsing '%s' failed: unable to parse major device id, err:%v", mm[0], err)
		}
		minor, err := strconv.Atoi(mm[1])
		if err != nil {
			return nil, fmt.Errorf("parsing '%s' failed: unable to parse minor device id, err:%v", mm[1], err)
		}

		info := MountInfo{
			ID:           id,
			ParentID:     parentID,
			Major:        major,
			Minor:        minor,
			Root:         fields[3],
			MountPoint:   fields[4],
			MountOptions: splitMountOptions(fields[5]),
		}
		// All fields until "-" are "optional fields".
		i := 6
		for ; i < len(fields) && fields[i] != "-"; i++ {
			info.OptionalFields = append(info.OptionalFields, fields[i])
		}
		// Parse the rest 3 fields.
		i++
		if len(fields)-i < 3 {
			return nil, fmt.Errorf("expect 3 fields in %s, got %d", line, len(fields)-i)
		}
		info.FsType = fields[i]
		info.Source = fields[i+1]
		info.SuperOptions = splitMountOptions(fields[i+2])
		infos = append(infos, info)
	}
	return infos, nil
}

// splitMountOptions parses comma-separated list of mount options into an array.
// It respects double quotes - commas in them are not considered as the option separator.
func splitMountOptions(s string) []string {
	inQuotes := false
	list := strings.FieldsFunc(s, func(r rune) bool {
		if r == '"' {
			inQuotes = !inQuotes
		}
		// Report a new field only when outside of double quotes.
		return r == ',' && !inQuotes
	})
	return list
}

// isMountPointMatch returns true if the path in mp is the same as dir.
// Handles case where mountpoint dir has been renamed due to stale NFS mount.
func isMountPointMatch(mp MountPoint, dir string) bool {
	deletedDir := fmt.Sprintf("%s\\040(deleted)", dir)
	return ((mp.Path == dir) || (mp.Path == deletedDir))
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
