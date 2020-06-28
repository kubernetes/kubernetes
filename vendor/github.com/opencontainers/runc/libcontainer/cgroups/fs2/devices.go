// +build linux

package fs2

import (
	"github.com/opencontainers/runc/libcontainer/cgroups/ebpf"
	"github.com/opencontainers/runc/libcontainer/cgroups/ebpf/devicefilter"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

func isRWM(perms configs.DevicePermissions) bool {
	var r, w, m bool
	for _, perm := range perms {
		switch perm {
		case 'r':
			r = true
		case 'w':
			w = true
		case 'm':
			m = true
		}
	}
	return r && w && m
}

// the logic is from crun
// https://github.com/containers/crun/blob/0.10.2/src/libcrun/cgroup.c#L1644-L1652
func canSkipEBPFError(cgroup *configs.Cgroup) bool {
	for _, dev := range cgroup.Resources.Devices {
		if dev.Allow || !isRWM(dev.Permissions) {
			return false
		}
	}
	return true
}

func setDevices(dirPath string, cgroup *configs.Cgroup) error {
	// XXX: This is currently a white-list (but all callers pass a blacklist of
	//      devices). This is bad for a whole variety of reasons, but will need
	//      to be fixed with co-ordinated effort with downstreams.
	devices := cgroup.Devices
	insts, license, err := devicefilter.DeviceFilter(devices)
	if err != nil {
		return err
	}
	dirFD, err := unix.Open(dirPath, unix.O_DIRECTORY|unix.O_RDONLY, 0600)
	if err != nil {
		return errors.Errorf("cannot get dir FD for %s", dirPath)
	}
	defer unix.Close(dirFD)
	// XXX: This code is currently incorrect when it comes to updating an
	//      existing cgroup with new rules (new rulesets are just appended to
	//      the program list because this uses BPF_F_ALLOW_MULTI). If we didn't
	//      use BPF_F_ALLOW_MULTI we could actually atomically swap the
	//      programs.
	//
	//      The real issue is that BPF_F_ALLOW_MULTI makes it hard to have a
	//      race-free blacklist because it acts as a whitelist by default, and
	//      having a deny-everything program cannot be overriden by other
	//      programs. You could temporarily insert a deny-everything program
	//      but that would result in spurrious failures during updates.
	if _, err := ebpf.LoadAttachCgroupDeviceFilter(insts, license, dirFD); err != nil {
		if !canSkipEBPFError(cgroup) {
			return err
		}
	}
	return nil
}
