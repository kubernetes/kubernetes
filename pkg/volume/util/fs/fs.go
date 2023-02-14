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

type UsageInfo struct {
	Bytes  int64
	Inodes int64
}

// FSInfo is the filesystem information
type FSInfo struct {
	// Available is the amount of space available on the filesystem in bytes
	Available int64
	// Capacity is the total capacity of the filesystem in bytes
	Capacity int64
	// Usage is the amount of used space on the filesystem in bytes
	Usage int64
	// Inodes is the amount of inodes available on the filesystem
	Inodes int64
	// InodesFree is the amount of inodes free for use on the disk
	InodesFree int64
	// InodesFree is the amount of inodes in use on the disk
	InodesUsed int64
	// ReadOnly indicates that the filesystem is mounted as read-only
	ReadOnly bool
}
