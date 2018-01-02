package platforms

import (
	"runtime"
	"strings"
)

// These function are generated from from https://golang.org/src/go/build/syslist.go.
//
// We use switch statements because they are slightly faster than map lookups
// and use a little less memory.

// isKnownOS returns true if we know about the operating system.
//
// The OS value should be normalized before calling this function.
func isKnownOS(os string) bool {
	switch os {
	case "android", "darwin", "dragonfly", "freebsd", "linux", "nacl", "netbsd", "openbsd", "plan9", "solaris", "windows", "zos":
		return true
	}
	return false
}

// isKnownArch returns true if we know about the architecture.
//
// The arch value should be normalized before being passed to this function.
func isKnownArch(arch string) bool {
	switch arch {
	case "386", "amd64", "amd64p32", "arm", "armbe", "arm64", "arm64be", "ppc64", "ppc64le", "mips", "mipsle", "mips64", "mips64le", "mips64p32", "mips64p32le", "ppc", "s390", "s390x", "sparc", "sparc64":
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
	case "aarch64":
		arch = "arm64"
		variant = "" // v8 is implied
	case "armhf":
		arch = "arm"
		variant = ""
	case "armel":
		arch = "arm"
		variant = "v6"
	case "arm":
		switch variant {
		case "v7", "7":
			variant = "v7"
		case "5", "6", "8":
			variant = "v" + variant
		}
	}

	return arch, variant
}
