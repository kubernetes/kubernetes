package system

import (
	"fmt"
	"runtime"
	"strings"

	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

// ValidatePlatform determines if a platform structure is valid.
// TODO This is a temporary function - can be replaced by parsing from
// https://github.com/containerd/containerd/pull/1403/files at a later date.
// @jhowardmsft
func ValidatePlatform(platform *specs.Platform) error {
	platform.Architecture = strings.ToLower(platform.Architecture)
	platform.OS = strings.ToLower(platform.OS)
	// Based on https://github.com/moby/moby/pull/34642#issuecomment-330375350, do
	// not support anything except operating system.
	if platform.Architecture != "" {
		return fmt.Errorf("invalid platform architecture %q", platform.Architecture)
	}
	if platform.OS != "" {
		if !(platform.OS == runtime.GOOS || (LCOWSupported() && platform.OS == "linux")) {
			return fmt.Errorf("invalid platform os %q", platform.OS)
		}
	}
	if len(platform.OSFeatures) != 0 {
		return fmt.Errorf("invalid platform osfeatures %q", platform.OSFeatures)
	}
	if platform.OSVersion != "" {
		return fmt.Errorf("invalid platform osversion %q", platform.OSVersion)
	}
	if platform.Variant != "" {
		return fmt.Errorf("invalid platform variant %q", platform.Variant)
	}
	return nil
}

// ParsePlatform parses a platform string in the format os[/arch[/variant]
// into an OCI image-spec platform structure.
// TODO This is a temporary function - can be replaced by parsing from
// https://github.com/containerd/containerd/pull/1403/files at a later date.
// @jhowardmsft
func ParsePlatform(in string) *specs.Platform {
	p := &specs.Platform{}
	elements := strings.SplitN(strings.ToLower(in), "/", 3)
	if len(elements) == 3 {
		p.Variant = elements[2]
	}
	if len(elements) >= 2 {
		p.Architecture = elements[1]
	}
	if len(elements) >= 1 {
		p.OS = elements[0]
	}
	return p
}
