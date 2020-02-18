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
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"

	"golang.org/x/sys/unix"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/volume/util/fsquota"
)

// FSInfo linux returns (available bytes, byte capacity, byte usage, total inodes, inodes free, inode usage, error)
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

const (
	// The block size in bytes.
	statBlockSize uint64 = 512
)

// DiskUsage gets disk usage of specified path.
func DiskUsage(dir string) (*resource.Quantity, error) {
	var bytes uint64

	if dir == "" {
		return nil, fmt.Errorf("invalid directory")
	}

	rootInfo, err := os.Stat(dir)
	if err != nil {
		return nil, fmt.Errorf("could not stat %q to get inode usage: %v", dir, err)
	}

	rootStat, ok := rootInfo.Sys().(*syscall.Stat_t)
	if !ok {
		return nil, fmt.Errorf("unsuported fileinfo for getting inode usage of %q", dir)
	}

	rootDevId := rootStat.Dev

	// dedupedInode stores inodes that could be duplicates (nlink > 1)
	dedupedInodes := make(map[uint64]struct{})

	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if os.IsNotExist(err) {
			// expected if files appear/vanish
			return nil
		}
		if err != nil {
			return fmt.Errorf("unable to count inodes for part of dir %s: %s", dir, err)
		}

		// according to the docs, Sys can be nil
		if info.Sys() == nil {
			return fmt.Errorf("fileinfo Sys is nil")
		}

		s, ok := info.Sys().(*syscall.Stat_t)
		if !ok {
			return fmt.Errorf("unsupported fileinfo; could not convert to stat_t")
		}

		if s.Dev != rootDevId {
			// don't descend into directories on other devices
			return filepath.SkipDir
		}
		if s.Nlink > 1 {
			if _, ok := dedupedInodes[s.Ino]; !ok {
				// Dedupe things that could be hardlinks
				dedupedInodes[s.Ino] = struct{}{}

				bytes += uint64(s.Blocks) * statBlockSize
			}
		} else {
			bytes += uint64(s.Blocks) * statBlockSize
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get disk usage due to error %v", err)
	}
	used, err := resource.ParseQuantity(strconv.FormatInt(int64(bytes), 10))
	if err != nil {
		return nil, fmt.Errorf("failed to get disk usage due to error %v", err)
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
	var counter byteCounter
	var stderr bytes.Buffer
	findCmd := exec.Command("find", path, "-xdev", "-printf", ".")
	findCmd.Stdout, findCmd.Stderr = &counter, &stderr
	if err := findCmd.Start(); err != nil {
		return 0, fmt.Errorf("failed to exec cmd %v - %v; stderr: %v", findCmd.Args, err, stderr.String())
	}
	if err := findCmd.Wait(); err != nil {
		return 0, fmt.Errorf("cmd %v failed. stderr: %s; err: %v", findCmd.Args, stderr.String(), err)
	}
	return counter.bytesWritten, nil
}

// Simple io.Writer implementation that counts how many bytes were written.
type byteCounter struct{ bytesWritten int64 }

func (b *byteCounter) Write(p []byte) (int, error) {
	b.bytesWritten += int64(len(p))
	return len(p), nil
}
