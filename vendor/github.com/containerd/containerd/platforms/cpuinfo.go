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

package platforms

import (
	"bufio"
	"os"
	"runtime"
	"strings"

	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/log"
	"github.com/pkg/errors"
)

// Present the ARM instruction set architecture, eg: v7, v8
var cpuVariant string

func init() {
	if isArmArch(runtime.GOARCH) {
		cpuVariant = getCPUVariant()
	} else {
		cpuVariant = ""
	}
}

// For Linux, the kernel has already detected the ABI, ISA and Features.
// So we don't need to access the ARM registers to detect platform information
// by ourselves. We can just parse these information from /proc/cpuinfo
func getCPUInfo(pattern string) (info string, err error) {
	if !isLinuxOS(runtime.GOOS) {
		return "", errors.Wrapf(errdefs.ErrNotImplemented, "getCPUInfo for OS %s", runtime.GOOS)
	}

	cpuinfo, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return "", err
	}
	defer cpuinfo.Close()

	// Start to Parse the Cpuinfo line by line. For SMP SoC, we parse
	// the first core is enough.
	scanner := bufio.NewScanner(cpuinfo)
	for scanner.Scan() {
		newline := scanner.Text()
		list := strings.Split(newline, ":")

		if len(list) > 1 && strings.EqualFold(strings.TrimSpace(list[0]), pattern) {
			return strings.TrimSpace(list[1]), nil
		}
	}

	// Check whether the scanner encountered errors
	err = scanner.Err()
	if err != nil {
		return "", err
	}

	return "", errors.Wrapf(errdefs.ErrNotFound, "getCPUInfo for pattern: %s", pattern)
}

func getCPUVariant() string {
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" {
		// Windows/Darwin only supports v7 for ARM32 and v8 for ARM64 and so we can use
		// runtime.GOARCH to determine the variants
		var variant string
		switch runtime.GOARCH {
		case "arm64":
			variant = "v8"
		case "arm":
			variant = "v7"
		default:
			variant = "unknown"
		}

		return variant
	}

	variant, err := getCPUInfo("Cpu architecture")
	if err != nil {
		log.L.WithError(err).Error("failure getting variant")
		return ""
	}

	switch strings.ToLower(variant) {
	case "8", "aarch64":
		// special case: if running a 32-bit userspace on aarch64, the variant should be "v7"
		if runtime.GOARCH == "arm" {
			variant = "v7"
		} else {
			variant = "v8"
		}
	case "7", "7m", "?(12)", "?(13)", "?(14)", "?(15)", "?(16)", "?(17)":
		variant = "v7"
	case "6", "6tej":
		variant = "v6"
	case "5", "5t", "5te", "5tej":
		variant = "v5"
	case "4", "4t":
		variant = "v4"
	case "3":
		variant = "v3"
	default:
		variant = "unknown"
	}

	return variant
}
