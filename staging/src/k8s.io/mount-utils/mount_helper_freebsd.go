//go:build freebsd
// +build freebsd

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

// MountInfo is a pure copy of the linux one, for now
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

// ParseMountInfo TODO: implement if needed, for FreeBSD (there is mountinfo, let's see)
func ParseMountInfo(filename string) ([]MountInfo, error) {
	
	infos := []MountInfo{}

	return infos, nil
}