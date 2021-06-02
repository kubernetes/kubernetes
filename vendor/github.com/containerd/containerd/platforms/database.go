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
	"runtime"
	"strings"
)

// isLinuxOS returns true if the operating system is Linux.
//
// The OS value should be normalized before calling this function.
func isLinuxOS(os string) bool {
	return os == "linux"
}

// These function are generated from https://golang.org/src/go/build/syslist.go.
//
// We use switch statements because they are slightly faster than map lookups
// and use a little less memory.

// isKnownOS returns true if we know about the operating system.
//
// The OS value should be normalized before calling this function.
func isKnownOS(os string) bool {
	switch os {
	case "aix", "android", "darwin", "dragonfly", "freebsd", "hurd", "illumos", "js", "linux", "nacl", "netbsd", "openbsd", "plan9", "solaris", "windows", "zos":
		return true
	}
	return false
}

// isArmArch returns true if the architecture is ARM.
//
// The arch value should be normalized before being passed to this function.
func isArmArch(arch string) bool {
	switch arch {
	case "arm", "arm64":
		return true
	}
	return false
}

// isKnownArch returns true if we know about the architecture.
//
// The arch value should be normalized before being passed to this function.
func isKnownArch(arch string) bool {
	switch arch {
	case "386", "amd64", "amd64p32", "arm", "armbe", "arm64", "arm64be", "ppc64", "ppc64le", "mips", "mipsle", "mips64", "mips64le", "mips64p32", "mips64p32le", "ppc", "riscv", "riscv64", "s390", "s390x", "sparc", "sparc64", "wasm":
		return true
	}
	return false
}

func normalizeOS(os string) string {
	if os == "" {
		return runtime.GOOS
	}
	os = strings.ToLower(os)

	switch os {
	case "macos":
		os = "darwin"
	}
	return os
}

// normalizeArch normalizes the architecture.
func normalizeArch(arch, variant string) (string, string) {
	arch, variant = strings.ToLower(arch), strings.ToLower(variant)
	switch arch {
	case "i386":
		arch = "386"
		variant = ""
	case "x86_64", "x86-64":
		arch = "amd64"
		variant = ""
	case "aarch64", "arm64":
		arch = "arm64"
		switch variant {
		case "8", "v8":
			variant = ""
		}
	case "armhf":
		arch = "arm"
		variant = "v7"
	case "armel":
		arch = "arm"
		variant = "v6"
	case "arm":
		switch variant {
		case "", "7":
			variant = "v7"
		case "5", "6", "8":
			variant = "v" + variant
		}
	}

	return arch, variant
}
