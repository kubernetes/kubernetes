// +build linux

package blkio

import (
	"errors"
	"fmt"
	"os"
	"syscall"

	"github.com/opencontainers/runc/libcontainer/configs"
	"golang.org/x/sys/unix"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

const (
	DeviceRoot = "/dev/mapper"
)

var (
	ErrNotADevice = errors.New("not a device node")

	unixos kubecontainer.OSInterface = &kubecontainer.RealOS{}
)

// DeviceFromPath given the path to a device and its cgroup_permissions(which cannot be easily queried)
// look up the information about a linux device and return that information as a Device struct.
func DeviceFromPath(path, permissions string) (device *configs.Device, err error) {
	defer func() {
		if e := recover(); e != nil {
			err = fmt.Errorf("%s", e)
		}
	}()
	var stat *syscall.Stat_t
	// var stat *os.fileStat
	fsinfo, err := unixos.Stat(path)
	if err != nil {
		return nil, err
	}
	stat = fsinfo.Sys().(*syscall.Stat_t)
	var (
		devNumber = stat.Rdev
		major     = unix.Major(devNumber)
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
		Major:       int64(major),
		Minor:       int64(unix.Minor(devNumber)),
		Permissions: permissions,
		FileMode:    os.FileMode(mode),
		Uid:         stat.Uid,
		Gid:         stat.Gid,
	}, nil
}
