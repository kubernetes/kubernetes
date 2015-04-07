// +build !linux

package nodes

import (
	"errors"

	"github.com/docker/libcontainer/devices"
)

func CreateDeviceNodes(rootfs string, nodesToCreate []*devices.Device) error {
	return errors.New("Unsupported method")
}
