// +build !have_btrfs

package btrfs

import (
	"errors"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/volume"
)

const (
	Name      = "btrfs"
	Type      = api.DriverType_DRIVER_TYPE_FILE
	RootParam = "home"
)

var (
	errUnsupported = errors.New("btrfs not supported on this platform")
)

func Init(params map[string]string) (volume.VolumeDriver, error) {
	return nil, errUnsupported
}
