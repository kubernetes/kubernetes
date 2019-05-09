// +build !linux

/*
Copyright 2017 The Kubernetes Authors.

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

package exec

import (
	"errors"

	"k8s.io/kubernetes/pkg/util/mount"
)

type execMounter struct{}

var _ = mount.Interface(&execMounter{})

// NewExecMounter returns a mounter that uses provided Exec interface to mount and
// unmount a filesystem. For all other calls it uses a wrapped mounter.
func NewExecMounter(exec mount.Exec, wrapped mount.Interface) mount.Interface {
	return &execMounter{}
}

func (mounter *execMounter) Mount(source string, target string, fstype string, options []string) error {
	return nil
}

func (mounter *execMounter) Unmount(target string) error {
	return nil
}

func (mounter *execMounter) List() ([]mount.MountPoint, error) {
	return []mount.MountPoint{}, nil
}

func (mounter *execMounter) IsMountPointMatch(mp mount.MountPoint, dir string) bool {
	return (mp.Path == dir)
}

func (mounter *execMounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return true, nil
}

func (mounter *execMounter) GetMountRefs(pathname string) ([]string, error) {
	return nil, errors.New("not implemented")
}
