package devices

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"
)

const (
	Wildcard = -1
)

var (
	ErrNotADeviceNode = errors.New("not a device node")
)

// Testing dependencies
var (
	osLstat       = os.Lstat
	ioutilReadDir = ioutil.ReadDir
)

type Device struct {
	Type              rune        `json:"type,omitempty"`
	Path              string      `json:"path,omitempty"`               // It is fine if this is an empty string in the case that you are using Wildcards
	MajorNumber       int64       `json:"major_number,omitempty"`       // Use the wildcard constant for wildcards.
	MinorNumber       int64       `json:"minor_number,omitempty"`       // Use the wildcard constant for wildcards.
	CgroupPermissions string      `json:"cgroup_permissions,omitempty"` // Typically just "rwm"
	FileMode          os.FileMode `json:"file_mode,omitempty"`          // The permission bits of the file's mode
	Uid               uint32      `json:"uid,omitempty"`
	Gid               uint32      `json:"gid,omitempty"`
}

func GetDeviceNumberString(deviceNumber int64) string {
	if deviceNumber == Wildcard {
		return "*"
	} else {
		return fmt.Sprintf("%d", deviceNumber)
	}
}

func (device *Device) GetCgroupAllowString() string {
	return fmt.Sprintf("%c %s:%s %s", device.Type, GetDeviceNumberString(device.MajorNumber), GetDeviceNumberString(device.MinorNumber), device.CgroupPermissions)
}

// Given the path to a device and it's cgroup_permissions(which cannot be easilly queried) look up the information about a linux device and return that information as a Device struct.
func GetDevice(path, cgroupPermissions string) (*Device, error) {
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
		return nil, ErrNotADeviceNode
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

	return &Device{
		Type:              devType,
		Path:              path,
		MajorNumber:       Major(devNumber),
		MinorNumber:       Minor(devNumber),
		CgroupPermissions: cgroupPermissions,
		FileMode:          fileModePermissionBits,
		Uid:               stat_t.Uid,
		Gid:               stat_t.Gid,
	}, nil
}

func GetHostDeviceNodes() ([]*Device, error) {
	return getDeviceNodes("/dev")
}

func getDeviceNodes(path string) ([]*Device, error) {
	files, err := ioutilReadDir(path)
	if err != nil {
		return nil, err
	}

	out := []*Device{}
	for _, f := range files {
		switch {
		case f.IsDir():
			switch f.Name() {
			case "pts", "shm", "fd", "mqueue":
				continue
			default:
				sub, err := getDeviceNodes(filepath.Join(path, f.Name()))
				if err != nil {
					return nil, err
				}

				out = append(out, sub...)
				continue
			}
		case f.Name() == "console":
			continue
		}

		device, err := GetDevice(filepath.Join(path, f.Name()), "rwm")
		if err != nil {
			if err == ErrNotADeviceNode {
				continue
			}
			return nil, err
		}
		out = append(out, device)
	}

	return out, nil
}
