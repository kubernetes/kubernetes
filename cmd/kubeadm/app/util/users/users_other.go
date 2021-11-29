//go:build !linux
// +build !linux

/*
Copyright 2021 The Kubernetes Authors.

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

package users

// EntryMap is empty on non-Linux.
type EntryMap struct{}

// UsersAndGroups is empty on non-Linux.
type UsersAndGroups struct{}

// ID is a NO-OP on non-Linux.
func (*EntryMap) ID(string) *int64 {
	return nil
}

// String is NO-OP on non-Linux.
func (*EntryMap) String() string {
	return ""
}

// AddUsersAndGroups is a NO-OP on non-Linux.
func AddUsersAndGroups() (*UsersAndGroups, error) {
	return nil, nil
}

// RemoveUsersAndGroups is a NO-OP on non-Linux.
func RemoveUsersAndGroups() error {
	return nil
}

// UpdatePathOwnerAndPermissions is a NO-OP on non-Linux.
func UpdatePathOwnerAndPermissions(path string, uid, gid int64, perms uint32) error {
	return nil
}

// UpdatePathOwner is a NO-OP on non-Linux.
func UpdatePathOwner(dirPath string, uid, gid int64) error {
	return nil
}
