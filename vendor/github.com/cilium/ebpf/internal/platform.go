package internal

import (
	"runtime"
)

// PlatformPrefix returns the platform-dependent syscall wrapper prefix used by
// the linux kernel.
//
// Based on https://github.com/golang/go/blob/master/src/go/build/syslist.go
// and https://github.com/libbpf/libbpf/blob/master/src/libbpf.c#L10047
func PlatformPrefix() string {
	switch runtime.GOARCH {
	case "386":
		return "__ia32_"
	case "amd64", "amd64p32":
		return "__x64_"

	case "arm", "armbe":
		return "__arm_"
	case "arm64", "arm64be":
		return "__arm64_"

	case "mips", "mipsle", "mips64", "mips64le", "mips64p32", "mips64p32le":
		return "__mips_"

	case "s390":
		return "__s390_"
	case "s390x":
		return "__s390x_"

	case "riscv", "riscv64":
		return "__riscv_"

	case "ppc":
		return "__powerpc_"
	case "ppc64", "ppc64le":
		return "__powerpc64_"

	default:
		return ""
	}
}
