/*
   Copyright The containerd Authors.

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

package cgroups

import (
	"errors"
	"os"
)

var (
	ErrInvalidPid               = errors.New("cgroups: pid must be greater than 0")
	ErrMountPointNotExist       = errors.New("cgroups: cgroup mountpoint does not exist")
	ErrInvalidFormat            = errors.New("cgroups: parsing file with invalid format failed")
	ErrFreezerNotSupported      = errors.New("cgroups: freezer cgroup not supported on this system")
	ErrMemoryNotSupported       = errors.New("cgroups: memory cgroup not supported on this system")
	ErrCgroupDeleted            = errors.New("cgroups: cgroup deleted")
	ErrNoCgroupMountDestination = errors.New("cgroups: cannot find cgroup mount destination")
)

// ErrorHandler is a function that handles and acts on errors
type ErrorHandler func(err error) error

// IgnoreNotExist ignores any errors that are for not existing files
func IgnoreNotExist(err error) error {
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

func errPassthrough(err error) error {
	return err
}
