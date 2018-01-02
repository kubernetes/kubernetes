// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//+build linux

package common

import (
	"errors"
	"fmt"
	"strings"

	stage1commontypes "github.com/coreos/rkt/stage1/common/types"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/unit"
)

var (
	ErrTooManySeccompIsolators = errors.New("too many seccomp isolators specified")
)

// Systemd filter mode, see
// https://www.freedesktop.org/software/systemd/man/systemd.exec.html#SystemCallFilter=
const (
	sdBlacklistPrefix = "~"
	sdWhitelistPrefix = ""
)

type filterType int

const (
	ModeBlacklist filterType = iota
	ModeWhitelist
)

// seccompFilter is an internal representation of the seccomp filtering
// supplied by the isolators.
type seccompFilter struct {
	syscalls             []string   // List of syscalls to filter
	mode                 filterType // whitelist or blacklist
	errno                string     // optional - empty string = use default
	forceNoNewPrivileges bool       // If true, then override the NoNewPrivileges isolator
}

// generateSeccompFilter computes the concrete seccomp filter from the isolators
func generateSeccompFilter(p *stage1commontypes.Pod, pa *preparedApp) (*seccompFilter, error) {
	sf := seccompFilter{}
	seenIsolators := 0
	for _, i := range pa.app.App.Isolators {
		var flag string
		var err error
		if seccomp, ok := i.Value().(types.LinuxSeccompSet); ok {
			seenIsolators++
			// By appc spec, only one seccomp isolator per app is allowed
			if seenIsolators > 1 {
				return nil, ErrTooManySeccompIsolators
			}
			switch i.Name {
			case types.LinuxSeccompRemoveSetName:
				sf.mode = ModeBlacklist
				sf.syscalls, flag, err = parseLinuxSeccompSet(p, seccomp)
				if err != nil {
					return nil, err
				}
				if flag == "empty" {
					// we interpret "remove @empty" to mean "default whitelist"
					sf.mode = ModeWhitelist
					sf.syscalls = RktDefaultSeccompWhitelist
				}
			case types.LinuxSeccompRetainSetName:
				sf.mode = ModeWhitelist
				sf.syscalls, flag, err = parseLinuxSeccompSet(p, seccomp)
				if err != nil {
					return nil, err
				}
				if flag == "all" {
					// Opt-out seccomp filtering
					return nil, nil
				}
			}
			sf.errno = string(seccomp.Errno())
		}
	}

	// If unset, use rkt default whitelist
	if seenIsolators == 0 {
		sf.mode = ModeWhitelist
		sf.syscalls = RktDefaultSeccompWhitelist
	}

	// Non-priv apps *must* have NoNewPrivileges set if they have seccomp
	sf.forceNoNewPrivileges = (pa.uid != 0)

	return &sf, nil
}

// seccompUnitOptions converts a concrete seccomp filter to systemd unit options
func seccompUnitOptions(opts []*unit.UnitOption, sf *seccompFilter) ([]*unit.UnitOption, error) {
	if sf == nil {
		return opts, nil
	}
	if sf.errno != "" {
		opts = append(opts, unit.NewUnitOption("Service", "SystemCallErrorNumber", sf.errno))
	}

	var filterPrefix string
	switch sf.mode {
	case ModeWhitelist:
		filterPrefix = sdWhitelistPrefix
	case ModeBlacklist:
		filterPrefix = sdBlacklistPrefix
	default:
		return nil, fmt.Errorf("unkown filter mode %v", sf.mode)
	}

	// SystemCallFilter options are written down one entry per line, because
	// filtering sets may be quite large and overlong lines break unit serialization.
	opts = appendOptionsList(opts, "Service", "SystemCallFilter", filterPrefix, sf.syscalls...)
	return opts, nil
}

// parseLinuxSeccompSet gets an appc LinuxSeccompSet and returns an array
// of values suitable for systemd SystemCallFilter.
func parseLinuxSeccompSet(p *stage1commontypes.Pod, s types.LinuxSeccompSet) (syscallFilter []string, flag string, err error) {
	for _, item := range s.Set() {
		if item[0] == '@' {
			// Wildcards
			wildcard := strings.SplitN(string(item), "/", 2)
			if len(wildcard) != 2 {
				continue
			}
			scope := wildcard[0]
			name := wildcard[1]
			switch scope {
			case "@appc.io":
				// appc-reserved wildcards
				switch name {
				case "all":
					return nil, "all", nil
				case "empty":
					return nil, "empty", nil
				}
			case "@docker":
				// Docker-originated wildcards
				switch name {
				case "default-blacklist":
					syscallFilter = append(syscallFilter, DockerDefaultSeccompBlacklist...)
				case "default-whitelist":
					syscallFilter = append(syscallFilter, DockerDefaultSeccompWhitelist...)
				}
			case "@rkt":
				// Custom rkt wildcards
				switch name {
				case "default-blacklist":
					syscallFilter = append(syscallFilter, RktDefaultSeccompBlacklist...)
				case "default-whitelist":
					syscallFilter = append(syscallFilter, RktDefaultSeccompWhitelist...)
				}
			case "@systemd":
				// Custom systemd wildcards (systemd >= 231)
				_, systemdVersion, err := GetFlavor(p)
				if err != nil || systemdVersion < 231 {
					return nil, "", errors.New("Unsupported or unknown systemd version, seccomp groups need systemd >= v231")
				}
				switch name {
				case "clock":
					syscallFilter = append(syscallFilter, "@clock")
				case "default-whitelist":
					syscallFilter = append(syscallFilter, "@default")
				case "mount":
					syscallFilter = append(syscallFilter, "@mount")
				case "network-io":
					syscallFilter = append(syscallFilter, "@network-io")
				case "obsolete":
					syscallFilter = append(syscallFilter, "@obsolete")
				case "privileged":
					syscallFilter = append(syscallFilter, "@privileged")
				case "process":
					syscallFilter = append(syscallFilter, "@process")
				case "raw-io":
					syscallFilter = append(syscallFilter, "@raw-io")
				}
			}
		} else {
			// Plain syscall name
			syscallFilter = append(syscallFilter, string(item))
		}
	}
	return syscallFilter, "", nil
}
