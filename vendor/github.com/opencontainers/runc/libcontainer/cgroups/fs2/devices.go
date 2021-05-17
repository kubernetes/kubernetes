// +build linux

package fs2

import (
	"github.com/opencontainers/runc/libcontainer/cgroups/ebpf"
	"github.com/opencontainers/runc/libcontainer/cgroups/ebpf/devicefilter"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/devices"
	"github.com/opencontainers/runc/libcontainer/userns"

	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

func isRWM(perms devices.Permissions) bool {
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

// This is similar to the logic applied in crun for handling errors from bpf(2)
// <https://github.com/containers/crun/blob/0.17/src/libcrun/cgroup.c#L2438-L2470>.
func canSkipEBPFError(r *configs.Resources) bool {
	// If we're running in a user namespace we can ignore eBPF rules because we
	// usually cannot use bpf(2), as well as rootless containers usually don't
	// have the necessary privileges to mknod(2) device inodes or access
	// host-level instances (though ideally we would be blocking device access
	// for rootless containers anyway).
	if userns.RunningInUserNS() {
		return true
	}

	// We cannot ignore an eBPF load error if any rule if is a block rule or it
	// doesn't permit all access modes.
	//
	// NOTE: This will sometimes trigger in cases where access modes are split
	//       between different rules but to handle this correctly would require
	//       using ".../libcontainer/cgroup/devices".Emulator.
	for _, dev := range r.Devices {
		if !dev.Allow || !isRWM(dev.Permissions) {
			return false
		}
	}
	return true
}

func setDevices(dirPath string, r *configs.Resources) error {
	if r.SkipDevices {
		return nil
	}
	// XXX: This is currently a white-list (but all callers pass a blacklist of
	//      devices). This is bad for a whole variety of reasons, but will need
	//      to be fixed with co-ordinated effort with downstreams.
	insts, license, err := devicefilter.DeviceFilter(r.Devices)
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
	//      having a deny-everything program cannot be overridden by other
	//      programs. You could temporarily insert a deny-everything program
	//      but that would result in spurrious failures during updates.
	if _, err := ebpf.LoadAttachCgroupDeviceFilter(insts, license, dirFD); err != nil {
		if !canSkipEBPFError(r) {
			return err
		}
	}
	return nil
}
