/*
Copyright The Kubernetes Authors.

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

package deviceattribute

import (
	"io/fs"
	"os"
)

// machine is the bare minimum machine/host abstraction we need.
// we use a wrapping struct to enable the functional option pattern
type machine struct {
	sysfs fs.ReadLinkFS
}

// MachineModifier is the type of functional options
type MachineModifier func(*machine)

// WithFS replaces the host sysfs with a custom FS
func WithFS(sysfs fs.ReadLinkFS) MachineModifier {
	return func(mc *machine) {
		mc.sysfs = sysfs
	}
}

// WithFSFromRoot replaces the host sysfs with a FS whose root
// is the provided path. Like chroot, this is for convenience
// and testability and for usage within containers, not for security.
func WithFSFromRoot(root string) MachineModifier {
	return func(mc *machine) {
		mc.sysfs = os.DirFS(root).(fs.ReadLinkFS)
	}
}
