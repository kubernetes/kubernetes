package seccomp

import (
	"fmt"

	rspec "github.com/opencontainers/runtime-spec/specs-go"
)

// ParseArchitectureFlag takes the raw string passed with the --arch flag, parses it
// and updates the Seccomp config accordingly
func ParseArchitectureFlag(architectureArg string, config *rspec.LinuxSeccomp) error {
	correctedArch, err := parseArch(architectureArg)
	if err != nil {
		return err
	}

	shouldAppend := true
	for _, alreadySpecified := range config.Architectures {
		if correctedArch == alreadySpecified {
			shouldAppend = false
		}
	}
	if shouldAppend {
		config.Architectures = append(config.Architectures, correctedArch)
	}
	return nil
}

func parseArch(arch string) (rspec.Arch, error) {
	arches := map[string]rspec.Arch{
		"x86":         rspec.ArchX86,
		"amd64":       rspec.ArchX86_64,
		"x32":         rspec.ArchX32,
		"arm":         rspec.ArchARM,
		"arm64":       rspec.ArchAARCH64,
		"mips":        rspec.ArchMIPS,
		"mips64":      rspec.ArchMIPS64,
		"mips64n32":   rspec.ArchMIPS64N32,
		"mipsel":      rspec.ArchMIPSEL,
		"mipsel64":    rspec.ArchMIPSEL64,
		"mipsel64n32": rspec.ArchMIPSEL64N32,
		"parisc":      rspec.ArchPARISC,
		"parisc64":    rspec.ArchPARISC64,
		"ppc":         rspec.ArchPPC,
		"ppc64":       rspec.ArchPPC64,
		"ppc64le":     rspec.ArchPPC64LE,
		"s390":        rspec.ArchS390,
		"s390x":       rspec.ArchS390X,
	}
	a, ok := arches[arch]
	if !ok {
		return "", fmt.Errorf("unrecognized architecture: %s", arch)
	}
	return a, nil
}
