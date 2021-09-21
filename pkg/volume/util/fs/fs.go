//go:build linux || darwin
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
	"path/filepath"
	"syscall"

	"golang.org/x/sys/unix"

	"k8s.io/kubernetes/pkg/volume/util/fsquota"
)

type UsageInfo struct {
	Bytes  int64
	Inodes int64
}

// Info linux returns (available bytes, byte capacity, byte usage, total inodes, inodes free, inode usage, error)
// for the filesystem that path resides upon.
func Info(path string) (int64, int64, int64, int64, int64, int64, error) {
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

// DiskUsage calculates the number of inodes and disk usage for a given directory
func DiskUsage(path string) (UsageInfo, error) {
	var usage UsageInfo

	if path == "" {
		return usage, fmt.Errorf("invalid directory")
	}

	// First check whether the quota system knows about this directory
	// A nil quantity or error means that the path does not support quotas
	// or xfs_quota tool is missing and we should use other mechanisms.
	consumption, _ := fsquota.GetConsumption(path)
	if consumption != nil {
		usage.Bytes = consumption.Value()
	}

	inodes, _ := fsquota.GetInodes(path)
	if inodes != nil {
		usage.Inodes = inodes.Value()
	}

	if inodes != nil && consumption != nil {
		return usage, nil
	}

	topLevelStat := &unix.Stat_t{}
	err := unix.Stat(path, topLevelStat)
	if err != nil {
		return usage, err
	}

	// dedupedInode stores inodes that could be duplicates (nlink > 1)
	dedupedInodes := make(map[uint64]struct{})

	err = filepath.Walk(path, func(path string, info os.FileInfo, err error) error {
		// ignore files that have been deleted after directory was read
		if os.IsNotExist(err) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("unable to count inodes for %s: %s", path, err)
		}

		// according to the docs, Sys can be nil
		if info.Sys() == nil {
			return fmt.Errorf("fileinfo Sys is nil")
		}

		s, ok := info.Sys().(*syscall.Stat_t)
		if !ok {
			return fmt.Errorf("unsupported fileinfo; could not convert to stat_t")
		}

		if s.Dev != topLevelStat.Dev {
			// don't descend into directories on other devices
			return filepath.SkipDir
		}

		// Dedupe hardlinks
		if s.Nlink > 1 {
			if _, ok := dedupedInodes[s.Ino]; !ok {
				dedupedInodes[s.Ino] = struct{}{}
			} else {
				return nil
			}
		}

		if consumption == nil {
			usage.Bytes += int64(s.Blocks) * int64(512) // blocksize in bytes
		}

		if inodes == nil {
			usage.Inodes++
		}

		return nil
	})

	return usage, err
}
