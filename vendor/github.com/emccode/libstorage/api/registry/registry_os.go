package registry

import (
	"strings"

	"github.com/emccode/libstorage/api/types"
)

type odm struct {
	types.OSDriver
	types.Context
}

// NewOSDriverManager returns a new OS driver manager.
func NewOSDriverManager(
	d types.OSDriver) types.OSDriver {
	return &odm{OSDriver: d}
}

func (d *odm) Mounts(
	ctx types.Context,
	deviceName, mountPoint string,
	opts types.Store) ([]*types.MountInfo, error) {
	ctx = ctx.Join(d.Context)

	return d.OSDriver.Mounts(ctx.Join(d.Context), deviceName, mountPoint, opts)
}

func (d *odm) Mount(
	ctx types.Context,
	deviceName, mountPoint string,
	opts *types.DeviceMountOpts) error {
	ctx = ctx.Join(d.Context)

	return d.OSDriver.Mount(ctx.Join(d.Context), deviceName, mountPoint, opts)
}

func (d *odm) Unmount(
	ctx types.Context,
	mountPoint string,
	opts types.Store) error {

	return d.OSDriver.Unmount(ctx.Join(d.Context), mountPoint, opts)
}

func (d *odm) IsMounted(
	ctx types.Context,
	mountPoint string,
	opts types.Store) (bool, error) {

	return d.OSDriver.IsMounted(ctx.Join(d.Context), mountPoint, opts)
}

func (d *odm) Format(
	ctx types.Context,
	deviceName string,
	opts *types.DeviceFormatOpts) error {

	ctx = ctx.Join(d.Context)

	if strings.Contains(deviceName, ":") {
		return nil
	}
	return d.OSDriver.Format(ctx, deviceName, opts)
}
