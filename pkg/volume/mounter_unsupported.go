// +build !linux

/*
Copyright 2014 Google Inc. All rights reserved.

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

package volume

const MOUNT_MS_BIND = 0
const MOUNT_MS_RDONLY = 0

type DiskMounter struct{}

func (mounter *DiskMounter) Mount(source string, target string, fstype string, flags uintptr, data string) error {
	return nil
}

func (mounter *DiskMounter) Unmount(target string, flags int) error {
	return nil
}

func (mounter *DiskMounter) RefCount(PD Interface) (string, int, error) {
	return "", 0, nil
}
