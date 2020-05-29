// +build seccomp

// SPDX-License-Identifier: Apache-2.0

// Copyright 2013-2018 Docker, Inc.

package seccomp // import "github.com/seccomp/containers-golang"

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/opencontainers/runtime-spec/specs-go"
	libseccomp "github.com/seccomp/libseccomp-golang"
	"golang.org/x/sys/unix"
)

//go:generate go run -tags 'seccomp' generate.go

// GetDefaultProfile returns the default seccomp profile.
func GetDefaultProfile(rs *specs.Spec) (*specs.LinuxSeccomp, error) {
	return setupSeccomp(DefaultProfile(), rs)
}

// LoadProfile takes a json string and decodes the seccomp profile.
func LoadProfile(body string, rs *specs.Spec) (*specs.LinuxSeccomp, error) {
	var config Seccomp
	if err := json.Unmarshal([]byte(body), &config); err != nil {
		return nil, fmt.Errorf("decoding seccomp profile failed: %v", err)
	}
	return setupSeccomp(&config, rs)
}

// LoadProfileFromBytes takes a byte slice and decodes the seccomp profile.
func LoadProfileFromBytes(body []byte, rs *specs.Spec) (*specs.LinuxSeccomp, error) {
	config := &Seccomp{}
	if err := json.Unmarshal(body, config); err != nil {
		return nil, fmt.Errorf("decoding seccomp profile failed: %v", err)
	}
	return setupSeccomp(config, rs)
}

// LoadProfileFromConfig takes a Seccomp struct and a spec to retrieve a LinuxSeccomp
func LoadProfileFromConfig(config *Seccomp, specgen *specs.Spec) (*specs.LinuxSeccomp, error) {
	return setupSeccomp(config, specgen)
}

var nativeToSeccomp = map[string]Arch{
	"amd64":       ArchX86_64,
	"arm64":       ArchAARCH64,
	"mips64":      ArchMIPS64,
	"mips64n32":   ArchMIPS64N32,
	"mipsel64":    ArchMIPSEL64,
	"mipsel64n32": ArchMIPSEL64N32,
	"s390x":       ArchS390X,
}

// inSlice tests whether a string is contained in a slice of strings or not.
// Comparison is case sensitive
func inSlice(slice []string, s string) bool {
	for _, ss := range slice {
		if s == ss {
			return true
		}
	}
	return false
}

func setupSeccomp(config *Seccomp, rs *specs.Spec) (*specs.LinuxSeccomp, error) {
	if config == nil {
		return nil, nil
	}

	// No default action specified, no syscalls listed, assume seccomp disabled
	if config.DefaultAction == "" && len(config.Syscalls) == 0 {
		return nil, nil
	}

	newConfig := &specs.LinuxSeccomp{}

	var arch string
	var native, err = libseccomp.GetNativeArch()
	if err == nil {
		arch = native.String()
	}

	if len(config.Architectures) != 0 && len(config.ArchMap) != 0 {
		return nil, errors.New("'architectures' and 'archMap' were specified in the seccomp profile, use either 'architectures' or 'archMap'")
	}

	// if config.Architectures == 0 then libseccomp will figure out the architecture to use
	if len(config.Architectures) != 0 {
		for _, a := range config.Architectures {
			newConfig.Architectures = append(newConfig.Architectures, specs.Arch(a))
		}
	}

	if len(config.ArchMap) != 0 {
		for _, a := range config.ArchMap {
			seccompArch, ok := nativeToSeccomp[arch]
			if ok {
				if a.Arch == seccompArch {
					newConfig.Architectures = append(newConfig.Architectures, specs.Arch(a.Arch))
					for _, sa := range a.SubArches {
						newConfig.Architectures = append(newConfig.Architectures, specs.Arch(sa))
					}
					break
				}
			}
		}
	}

	newConfig.DefaultAction = specs.LinuxSeccompAction(config.DefaultAction)

Loop:
	// Loop through all syscall blocks and convert them to libcontainer format after filtering them
	for _, call := range config.Syscalls {
		if len(call.Excludes.Arches) > 0 {
			if inSlice(call.Excludes.Arches, arch) {
				continue Loop
			}
		}
		if len(call.Excludes.Caps) > 0 {
			for _, c := range call.Excludes.Caps {
				if inSlice(rs.Process.Capabilities.Bounding, c) {
					continue Loop
				}
			}
		}
		if len(call.Includes.Arches) > 0 {
			if !inSlice(call.Includes.Arches, arch) {
				continue Loop
			}
		}
		if len(call.Includes.Caps) > 0 {
			for _, c := range call.Includes.Caps {
				if !inSlice(rs.Process.Capabilities.Bounding, c) {
					continue Loop
				}
			}
		}

		if call.Name != "" && len(call.Names) != 0 {
			return nil, errors.New("'name' and 'names' were specified in the seccomp profile, use either 'name' or 'names'")
		}

		if call.Name != "" {
			newConfig.Syscalls = append(newConfig.Syscalls, createSpecsSyscall([]string{call.Name}, call.Action, call.Args, call.ErrnoRet))
		}

		if len(call.Names) > 0 {
			newConfig.Syscalls = append(newConfig.Syscalls, createSpecsSyscall(call.Names, call.Action, call.Args, call.ErrnoRet))
		}
	}

	return newConfig, nil
}

func createSpecsSyscall(names []string, action Action, args []*Arg, errnoRet *uint) specs.LinuxSyscall {
	newCall := specs.LinuxSyscall{
		Names:    names,
		Action:   specs.LinuxSeccompAction(action),
		ErrnoRet: errnoRet,
	}

	// Loop through all the arguments of the syscall and convert them
	for _, arg := range args {
		newArg := specs.LinuxSeccompArg{
			Index:    arg.Index,
			Value:    arg.Value,
			ValueTwo: arg.ValueTwo,
			Op:       specs.LinuxSeccompOperator(arg.Op),
		}

		newCall.Args = append(newCall.Args, newArg)
	}
	return newCall
}

// IsEnabled returns true if seccomp is enabled for the host.
func IsEnabled() bool {
	// Check if Seccomp is supported, via CONFIG_SECCOMP.
	if err := unix.Prctl(unix.PR_GET_SECCOMP, 0, 0, 0, 0); err != unix.EINVAL {
		// Make sure the kernel has CONFIG_SECCOMP_FILTER.
		if err := unix.Prctl(unix.PR_SET_SECCOMP, unix.SECCOMP_MODE_FILTER, 0, 0, 0); err != unix.EINVAL {
			return true
		}
	}
	return false
}
