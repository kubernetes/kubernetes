package devices

import (
	"errors"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/opencontainers/runc/libcontainer/configs"

	"golang.org/x/sys/unix"
)

var (
	ErrNotADevice = errors.New("not a device node")
)

// Testing dependencies
var (
	unixLstat     = unix.Lstat
	ioutilReadDir = ioutil.ReadDir
)

// Given the path to a device and its cgroup_permissions(which cannot be easily queried) look up the information about a linux device and return that information as a Device struct.
func DeviceFromPath(path, permissions string) (*configs.Device, error) {
	var stat unix.Stat_t
	err := unixLstat(path, &stat)
	if err != nil {
		return nil, err
	}

	var (
		devNumber = int(stat.Rdev)
		major     = Major(devNumber)
	)
	if major == 0 {
		return nil, ErrNotADevice
	}

	var (
		devType rune
		mode    = stat.Mode
	)
	switch {
	case mode&unix.S_IFBLK == unix.S_IFBLK:
		devType = 'b'
	case mode&unix.S_IFCHR == unix.S_IFCHR:
		devType = 'c'
	}
	return &configs.Device{
		Type:        devType,
		Path:        path,
		Major:       major,
		Minor:       Minor(devNumber),
		Permissions: permissions,
		FileMode:    os.FileMode(mode),
		Uid:         stat.Uid,
		Gid:         stat.Gid,
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
			// ".lxc" & ".lxd-mounts" added to address https://github.com/lxc/lxd/issues/2825
			case "pts", "shm", "fd", "mqueue", ".lxc", ".lxd-mounts":
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
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		out = append(out, device)
	}
	return out, nil
}
