// Copyright 2014 Google Inc. All Rights Reserved.
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

//go:build linux

package vfs

import (
	"context"
	"syscall"
	"time"
)

// GetVfsStats returns filesystem statistics using the statfs syscall.
// It has a timeout to prevent hanging on unresponsive filesystems.
func GetVfsStats(path string) (total uint64, free uint64, avail uint64, inodes uint64, inodesFree uint64, err error) {
	// timeout the context with, default is 2sec
	timeout := 2
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	type result struct {
		total      uint64
		free       uint64
		avail      uint64
		inodes     uint64
		inodesFree uint64
		err        error
	}

	resultChan := make(chan result, 1)

	go func() {
		var s syscall.Statfs_t
		if err = syscall.Statfs(path, &s); err != nil {
			total, free, avail, inodes, inodesFree = 0, 0, 0, 0, 0
		}
		total = uint64(s.Frsize) * s.Blocks
		free = uint64(s.Frsize) * s.Bfree
		avail = uint64(s.Frsize) * s.Bavail
		inodes = uint64(s.Files)
		inodesFree = uint64(s.Ffree)
		resultChan <- result{total: total, free: free, avail: avail, inodes: inodes, inodesFree: inodesFree, err: err}
	}()

	select {
	case <-ctx.Done():
		return 0, 0, 0, 0, 0, ctx.Err()
	case res := <-resultChan:
		return res.total, res.free, res.avail, res.inodes, res.inodesFree, res.err
	}
}
