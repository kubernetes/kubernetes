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

// TODO(thockin): This whole pkg is pretty linux-centric.  As soon as we have
// an alternate platform, we will need to abstract further.
package mount

// Each supported platform must define the following flags:
//   - FlagBind: specifies a bind mount
//   - FlagReadOnly: the mount will be read-only
type Interface interface {
	// Mount wraps syscall.Mount().
	Mount(source string, target string, fstype string, flags uintptr, data string) error

	// Umount wraps syscall.Mount().
	Unmount(target string, flags int) error

	// List returns a list of all mounted filesystems.  This can be large.
	// On some platforms, reading mounts is not guaranteed consistent (i.e.
	// it could change between chunked reads). This is guaranteed to be
	// consistent.
	List() ([]MountPoint, error)
}

// New returns a mount.Interface for the current system.
func New() Interface {
	return &Mounter{}
}

// This represents a single line in /proc/mounts or /etc/fstab.
type MountPoint struct {
	Device string
	Path   string
	Type   string
	Opts   []string
	Freq   int
	Pass   int
}

// Examines /proc/mounts to find all other references to the device referenced
// by mountPath.
func GetMountRefs(mounter Interface, mountPath string) ([]string, error) {
	mps, err := mounter.List()
	if err != nil {
		return nil, err
	}

	// Find the device name.
	deviceName := ""
	for i := range mps {
		if mps[i].Path == mountPath {
			deviceName = mps[i].Device
			break
		}
	}

	// Find all references to the device.
	var refs []string
	for i := range mps {
		if mps[i].Device == deviceName && mps[i].Path != mountPath {
			refs = append(refs, mps[i].Path)
		}
	}
	return refs, nil
}
