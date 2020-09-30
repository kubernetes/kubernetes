// +build linux darwin

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

package fs

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/volume/util/fsquota"
)

// FsInfo linux returns (available bytes, byte capacity, byte usage, total inodes, inodes free, inode usage, error)
// for the filesystem that path resides upon.
func FsInfo(path string) (int64, int64, int64, int64, int64, int64, error) {
	statfs := &unix.Statfs_t{}
	err := unix.Statfs(path, statfs)
	if err != nil {
		return 0, 0, 0, 0, 0, 0, err
	}

	// Available is blocks available * fragment size
	available := int64(statfs.Bavail) * int64(statfs.Bsize)

	// Capacity is total block count * fragment size
	capacity := int64(statfs.Blocks) * int64(statfs.Bsize)

	// Usage is block being used * fragment size (aka block size).
	usage := (int64(statfs.Blocks) - int64(statfs.Bfree)) * int64(statfs.Bsize)

	inodes := int64(statfs.Files)
	inodesFree := int64(statfs.Ffree)
	inodesUsed := inodes - inodesFree

	return available, capacity, usage, inodes, inodesFree, inodesUsed, nil
}

// DiskUsage gets disk usage of specified path.
func DiskUsage(path string) (*resource.Quantity, error) {
	// First check whether the quota system knows about this directory
	// A nil quantity with no error means that the path does not support quotas
	// and we should use other mechanisms.
	data, err := fsquota.GetConsumption(path)
	if data != nil {
		return data, nil
	} else if err != nil {
		return nil, fmt.Errorf("unable to retrieve disk consumption via quota for %s: %v", path, err)
	}
	// Uses the same niceness level as cadvisor.fs does when running du
	// Uses -B 1 to always scale to a blocksize of 1 byte
	out, err := exec.Command("nice", "-n", "19", "du", "-x", "-s", "-B", "1", path).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed command 'du' ($ nice -n 19 du -x -s -B 1) on path %s with error %v", path, err)
	}
	used, err := resource.ParseQuantity(strings.Fields(string(out))[0])
	if err != nil {
		return nil, fmt.Errorf("failed to parse 'du' output %s due to error %v", out, err)
	}
	used.Format = resource.BinarySI
	return &used, nil
}

// Find uses the equivalent of the command `find <path> -dev -printf '.' | wc -c` to count files and directories.
// While this is not an exact measure of inodes used, it is a very good approximation.
func Find(path string) (int64, error) {
	if path == "" {
		return 0, fmt.Errorf("invalid directory")
	}
	// First check whether the quota system knows about this directory
	// A nil quantity with no error means that the path does not support quotas
	// and we should use other mechanisms.
	inodes, err := fsquota.GetInodes(path)
	if inodes != nil {
		return inodes.Value(), nil
	} else if err != nil {
		return 0, fmt.Errorf("unable to retrieve inode consumption via quota for %s: %v", path, err)
	}

	topLevelStat := &unix.Stat_t{}
	err = unix.Stat(path, topLevelStat)
	if err != nil {
		return 0, err
	}

	var count int64 = 0
	err = filepath.Walk(path, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			// ignore files that have been deleted after directory was read
			if os.IsNotExist(err) {
				return nil
			}
			count = 0
			return err
		}
		count++
		if info.IsDir() {
			stat := &unix.Stat_t{}
			err = unix.Stat(path, stat)
			if err != nil {
				return nil
			}
			if stat.Dev != topLevelStat.Dev {
				return filepath.SkipDir
			}
		}
		return nil
	})
	return count, err
}
