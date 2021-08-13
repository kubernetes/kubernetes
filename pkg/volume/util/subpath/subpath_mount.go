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

// TODO(thockin): This whole pkg is pretty linux-centric.  As soon as we have
// an alternate platform, we will need to abstract further.

package subpath

import (
	"k8s.io/utils/mount"
)

// MountInterface defines the set of methods to allow for mount operations on a system.
type MountInterface interface {
	mount.Interface

	// MountSensitiveWithFlags is the same as MountSensitive() with additional mount flags
	MountSensitiveWithFlags(source string, target string, fstype string, options []string, sensitiveOptions []string, mountFlags []string) error
}

// Compile-time check to ensure all Mounter implementations satisfy
// the mount interface.
var _ MountInterface = &Mounter{}
