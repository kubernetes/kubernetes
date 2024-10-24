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

   File copied and customized based on
   https://github.com/moby/moby/tree/v20.10.14/profiles/seccomp/kernel_linux.go

   File copied from
   https://github.com/containerd/containerd/blob/v1.7.5/contrib/seccomp/kernelversion/kernel_linux.go
*/

package kernelversion

import (
	"bytes"
	"fmt"
	"sync"

	"golang.org/x/sys/unix"
)

// KernelVersion holds information about the kernel.
type KernelVersion struct {
	Kernel uint64 // Version of the Kernel (i.e., the "4" in "4.1.2-generic")
	Major  uint64 // Major revision of the Kernel (i.e., the "1" in "4.1.2-generic")
}

func (k *KernelVersion) String() string {
	if k.Kernel > 0 || k.Major > 0 {
		return fmt.Sprintf("%d.%d", k.Kernel, k.Major)
	}
	return ""
}

var (
	currentKernelVersion *KernelVersion
	kernelVersionError   error
	once                 sync.Once
)

// getKernelVersion gets the current kernel version.
func getKernelVersion() (*KernelVersion, error) {
	once.Do(func() {
		var uts unix.Utsname
		if err := unix.Uname(&uts); err != nil {
			return
		}
		// Remove the \x00 from the release for Atoi to parse correctly
		currentKernelVersion, kernelVersionError = parseRelease(string(uts.Release[:bytes.IndexByte(uts.Release[:], 0)]))
	})
	return currentKernelVersion, kernelVersionError
}

// parseRelease parses a string and creates a KernelVersion based on it.
func parseRelease(release string) (*KernelVersion, error) {
	var version KernelVersion

	// We're only make sure we get the "kernel" and "major revision". Sometimes we have
	// 3.12.25-gentoo, but sometimes we just have 3.12-1-amd64.
	_, err := fmt.Sscanf(release, "%d.%d", &version.Kernel, &version.Major)
	if err != nil {
		return nil, fmt.Errorf("failed to parse kernel version %q: %w", release, err)
	}
	return &version, nil
}

// GreaterEqualThan checks if the host's kernel version is greater than, or
// equal to the given kernel version v. Only "kernel version" and "major revision"
// can be specified (e.g., "3.12") and will be taken into account, which means
// that 3.12.25-gentoo and 3.12-1-amd64 are considered equal (kernel: 3, major: 12).
func GreaterEqualThan(minVersion KernelVersion) (bool, error) {
	kv, err := getKernelVersion()
	if err != nil {
		return false, err
	}
	if kv.Kernel > minVersion.Kernel {
		return true, nil
	}
	if kv.Kernel == minVersion.Kernel && kv.Major >= minVersion.Major {
		return true, nil
	}
	return false, nil
}
