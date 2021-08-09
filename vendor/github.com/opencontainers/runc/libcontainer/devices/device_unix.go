// +build !windows

package devices

import (
	"errors"
	"io/ioutil"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// ErrNotADevice denotes that a file is not a valid linux device.
var ErrNotADevice = errors.New("not a device node")

// Testing dependencies
var (
	unixLstat     = unix.Lstat
	ioutilReadDir = ioutil.ReadDir
)

func mkDev(d *Rule) (uint64, error) {
	if d.Major == Wildcard || d.Minor == Wildcard {
		return 0, errors.New("cannot mkdev() device with wildcards")
	}
	return unix.Mkdev(uint32(d.Major), uint32(d.Minor)), nil
}

// DeviceFromPath takes the path to a device and its cgroup_permissions (which
// cannot be easily queried) to look up the information about a linux device
// and returns that information as a Device struct.
func DeviceFromPath(path, permissions string) (*Device, error) {
	var stat unix.Stat_t
	err := unixLstat(path, &stat)
	if err != nil {
		return nil, err
	}

	var (
		devType   Type
		mode      = stat.Mode
		devNumber = uint64(stat.Rdev)
		major     = unix.Major(devNumber)
		minor     = unix.Minor(devNumber)
	)
	switch mode & unix.S_IFMT {
	case unix.S_IFBLK:
		devType = BlockDevice
	case unix.S_IFCHR:
		devType = CharDevice
	case unix.S_IFIFO:
		devType = FifoDevice
	default:
		return nil, ErrNotADevice
	}
	return &Device{
		Rule: Rule{
			Type:        devType,
			Major:       int64(major),
			Minor:       int64(minor),
			Permissions: Permissions(permissions),
		},
		Path:     path,
		FileMode: os.FileMode(mode &^ unix.S_IFMT),
		Uid:      stat.Uid,
		Gid:      stat.Gid,
	}, nil
}

// HostDevices returns all devices that can be found under /dev directory.
func HostDevices() ([]*Device, error) {
	return GetDevices("/dev")
}

// GetDevices recursively traverses a directory specified by path
// and returns all devices found there.
func GetDevices(path string) ([]*Device, error) {
	files, err := ioutilReadDir(path)
	if err != nil {
		return nil, err
	}
	var out []*Device
	for _, f := range files {
		switch {
		case f.IsDir():
			switch f.Name() {
			// ".lxc" & ".lxd-mounts" added to address https://github.com/lxc/lxd/issues/2825
			// ".udev" added to address https://github.com/opencontainers/runc/issues/2093
			case "pts", "shm", "fd", "mqueue", ".lxc", ".lxd-mounts", ".udev":
				continue
			default:
				sub, err := GetDevices(filepath.Join(path, f.Name()))
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
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		if device.Type == FifoDevice {
			continue
		}
		out = append(out, device)
	}
	return out, nil
}
