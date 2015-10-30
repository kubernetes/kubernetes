// +build linux freebsd

package devices

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"

	"github.com/docker/libcontainer/configs"
)

var (
	ErrNotADevice = errors.New("not a device node")
)

// Testing dependencies
var (
	osLstat       = os.Lstat
	ioutilReadDir = ioutil.ReadDir
)

// Given the path to a device and it's cgroup_permissions(which cannot be easily queried) look up the information about a linux device and return that information as a Device struct.
func DeviceFromPath(path, permissions string) (*configs.Device, error) {
	fileInfo, err := osLstat(path)
	if err != nil {
		return nil, err
	}
	var (
		devType                rune
		mode                   = fileInfo.Mode()
		fileModePermissionBits = os.FileMode.Perm(mode)
	)
	switch {
	case mode&os.ModeDevice == 0:
		return nil, ErrNotADevice
	case mode&os.ModeCharDevice != 0:
		fileModePermissionBits |= syscall.S_IFCHR
		devType = 'c'
	default:
		fileModePermissionBits |= syscall.S_IFBLK
		devType = 'b'
	}
	stat_t, ok := fileInfo.Sys().(*syscall.Stat_t)
	if !ok {
		return nil, fmt.Errorf("cannot determine the device number for device %s", path)
	}
	devNumber := int(stat_t.Rdev)
	return &configs.Device{
		Type:        devType,
		Path:        path,
		Major:       Major(devNumber),
		Minor:       Minor(devNumber),
		Permissions: permissions,
		FileMode:    fileModePermissionBits,
		Uid:         stat_t.Uid,
		Gid:         stat_t.Gid,
	}, nil
}

func HostDevices() ([]*configs.Device, error) {
	return getDevices("/dev")
}

func getDevices(path string) ([]*configs.Device, error) {
	files, err := ioutilReadDir(path)
	if err != nil {
		return nil, err
	}
	out := []*configs.Device{}
	for _, f := range files {
		switch {
		case f.IsDir():
			switch f.Name() {
			case "pts", "shm", "fd", "mqueue":
				continue
			default:
				sub, err := getDevices(filepath.Join(path, f.Name()))
				if err != nil {
					return nil, err
				}

				out = append(out, sub...)
				continue
			}
		case f.Name() == "console":
			continue
		}
		device, err := DeviceFromPath(filepath.Join(path, f.Name()), "rwm")
		if err != nil {
			if err == ErrNotADevice {
				continue
			}
			return nil, err
		}
		out = append(out, device)
	}
	return out, nil
}
