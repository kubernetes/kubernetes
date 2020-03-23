// +build linux

package fs2

import (
	"github.com/opencontainers/runc/libcontainer/cgroups/ebpf"
	"github.com/opencontainers/runc/libcontainer/cgroups/ebpf/devicefilter"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

func isRWM(cgroupPermissions string) bool {
	r := false
	w := false
	m := false
	for _, rn := range cgroupPermissions {
		switch rn {
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
	devices := cgroup.Devices
	if allowAllDevices := cgroup.Resources.AllowAllDevices; allowAllDevices != nil {
		// never set by OCI specconv, but *allowAllDevices=false is still used by the integration test
		if *allowAllDevices == true {
			return errors.New("libcontainer AllowAllDevices is not supported, use Devices")
		}
		for _, ad := range cgroup.Resources.AllowedDevices {
			d := *ad
			d.Allow = true
			devices = append(devices, &d)
		}
	}
	if len(cgroup.Resources.DeniedDevices) != 0 {
		// never set by OCI specconv
		return errors.New("libcontainer DeniedDevices is not supported, use Devices")
	}
	insts, license, err := devicefilter.DeviceFilter(devices)
	if err != nil {
		return err
	}
	dirFD, err := unix.Open(dirPath, unix.O_DIRECTORY|unix.O_RDONLY, 0600)
	if err != nil {
		return errors.Errorf("cannot get dir FD for %s", dirPath)
	}
	defer unix.Close(dirFD)
	if _, err := ebpf.LoadAttachCgroupDeviceFilter(insts, license, dirFD); err != nil {
		if !canSkipEBPFError(cgroup) {
			return err
		}
	}
	return nil
}
