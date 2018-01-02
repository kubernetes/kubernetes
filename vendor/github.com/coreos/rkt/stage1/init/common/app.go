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
	"os"
	"path/filepath"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/cgroup"
	stage1commontypes "github.com/coreos/rkt/stage1/common/types"
)

// preparedApp contains some internal state needed to actually run an app.
// We add this intermediate step to prevent unit file generation from being
// totally unwieldly.
type preparedApp struct {
	app             *schema.RuntimeApp
	uid             uint32
	gid             uint32
	env             types.Environment
	resources       appResources
	mounts          []Mount
	noNewPrivileges bool
	capabilities    []string
	seccomp         *seccompFilter

	// Path restrictions
	roPaths     []string
	hiddenPaths []string
	hiddenDirs  []string
}

type appResources struct {
	MemoryLimit         *uint64 // Memory limit in bytes
	CPUQuota            *uint64 // The hard (absolute) CPU quota as a percent (100 = 1 core)
	LinuxCPUShares      *uint64 // The relative CPU weight in the app's cgroup.
	LinuxOOMScoreAdjust *int    // OOMScoreAdjust knob
}

/*
 * Paths to protect for non-provileged applications
 * AKA protectKernelTunables
 */
var protectKernelROPaths = []string{
	"/proc/bus/",
	"/proc/sys/kernel/core_pattern",
	"/proc/sys/kernel/modprobe",
	"/proc/sys/vm/panic_on_oom",
	"/proc/sysrq-trigger",
	"/sys/block/",
	"/sys/bus/",
	"/sys/class/",
	"/sys/dev/",
	"/sys/devices/",
	"/sys/kernel/",
}
var protectKernelHiddenDirs = []string{
	"/sys/firmware/",
	"/sys/fs/",
	"/sys/hypervisor/",
	"/sys/module/",
	"/sys/power/",
}

// This is separate because systemd <231 didn't support masking files,
// only directories
var protectKernelHiddenPaths = []string{
	"/proc/config.gz",
	"/proc/kallsyms",
	"/proc/sched_debug",
	"/proc/kcore",
	"/proc/kmem",
	"/proc/mem",
}

// prepareApp sets up the internal runtime context for a specific app.
func prepareApp(p *stage1commontypes.Pod, ra *schema.RuntimeApp) (*preparedApp, error) {
	pa := preparedApp{
		app:             ra,
		env:             ra.App.Environment,
		noNewPrivileges: getAppNoNewPrivileges(ra.App.Isolators),
	}
	var err error

	// Determine numeric uid and gid
	u, g, err := parseUserGroup(p, ra)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to determine app's uid and gid"), err)
	}
	if u < 0 || g < 0 {
		return nil, errors.New("Invalid uid or gid")
	}
	pa.uid = uint32(u)
	pa.gid = uint32(g)

	// Set some rkt-provided environment variables
	pa.env.Set("AC_APP_NAME", ra.Name.String())
	if p.MetadataServiceURL != "" {
		pa.env.Set("AC_METADATA_URL", p.MetadataServiceURL)
	}

	// Determine capability set
	pa.capabilities, err = getAppCapabilities(ra.App.Isolators)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to construct capabilities"), err)
	}

	// Determine mounts
	cfd := ConvertedFromDocker(p.Images[ra.Name.String()])
	pa.mounts, err = GenerateMounts(ra, p.Manifest.Volumes, cfd)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to compute mounts"), err)
	}

	// Compute resources
	pa.resources, err = computeAppResources(ra.App.Isolators)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to compute resources"), err)
	}

	// Protect kernel tunables by default
	if !p.InsecureOptions.DisablePaths {
		pa.roPaths = append(pa.roPaths, protectKernelROPaths...)
		pa.hiddenPaths = append(pa.hiddenDirs, protectKernelHiddenPaths...)
		pa.hiddenDirs = append(pa.hiddenDirs, protectKernelHiddenDirs...)
	}

	// Seccomp
	if !p.InsecureOptions.DisableSeccomp {
		pa.seccomp, err = generateSeccompFilter(p, &pa)
		if err != nil {
			return nil, err
		}
		if pa.seccomp != nil && pa.seccomp.forceNoNewPrivileges {
			pa.noNewPrivileges = true
		}
	}

	// Write the systemd-sysusers config file
	if err := generateSysusers(p, pa.app, int(pa.uid), int(pa.gid), &p.UidRange); err != nil {
		return nil, errwrap.Wrapf("unable to generate sysusers file", err)
	}

	return &pa, nil
}

// computeAppResources processes any isolators that manipulate cgroups.
func computeAppResources(isolators types.Isolators) (appResources, error) {
	res := appResources{}
	var err error

	withIsolator := func(name string, f func() error) error {
		ok, err := cgroup.IsIsolatorSupported(name)
		if err != nil {
			return errwrap.Wrapf("could not check for isolator "+name, err)
		}

		if !ok {
			fmt.Fprintf(os.Stderr, "warning: resource/%s isolator set but support disabled in the kernel, skipping\n", name)
			return nil
		}

		return f()
	}

	for _, isolator := range isolators {
		if err != nil {
			return res, err
		}

		switch v := isolator.Value().(type) {
		case *types.ResourceMemory:
			err = withIsolator("memory", func() error {
				if v.Limit() == nil {
					return nil
				}

				val := uint64(v.Limit().Value())
				res.MemoryLimit = &val
				return nil
			})
		case *types.ResourceCPU:
			err = withIsolator("cpu", func() error {
				if v.Limit() == nil {
					return nil
				}
				if v.Limit().Value() > MaxMilliValue {
					return fmt.Errorf("cpu limit exceeds the maximum millivalue: %v", v.Limit().String())
				}

				val := uint64(v.Limit().MilliValue() / 10)
				res.CPUQuota = &val
				return nil
			})
		case *types.LinuxCPUShares:
			err = withIsolator("cpu", func() error {
				val := uint64(*v)
				res.LinuxCPUShares = &val
				return nil
			})
		case *types.LinuxOOMScoreAdj:
			val := int(*v)
			res.LinuxOOMScoreAdjust = &val
		}
	}

	return res, err
}

// relAppPaths prepends the relative app path (/opt/stage1/rootfs/) to a list
// of paths. Useful for systemd unit directives.
func (pa *preparedApp) relAppPaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		out = append(out, filepath.Join(common.RelAppRootfsPath(pa.app.Name), p))
	}
	return out
}
