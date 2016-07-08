package client

import (
	"os"
	"path"

	"github.com/akutz/gofig"

	"github.com/emccode/libstorage/api/registry"
	"github.com/emccode/libstorage/api/types"
	"github.com/emccode/libstorage/drivers/storage/vfs"
)

type driver struct {
	config gofig.Config
}

func init() {
	registry.RegisterClientDriver(vfs.Name, newDriver)
}

func newDriver() types.ClientDriver {
	return &driver{}
}

func (d *driver) Name() string {
	return vfs.Name
}

func (d *driver) Init(ctx types.Context, config gofig.Config) error {
	d.config = config
	os.MkdirAll(vfs.VolumesDirPath(config), 0755)
	return nil
}

func (d *driver) InstanceInspectBefore(ctx *types.Context) error {
	return nil
}

func (d *driver) InstanceInspectAfter(
	ctx types.Context, result *types.Instance) {
}

func (d *driver) VolumesBefore(ctx *types.Context) error {
	return nil
}

func (d *driver) VolumesAfter(
	ctx types.Context, result *types.ServiceVolumeMap) {
}

func (d *driver) VolumesByServiceBefore(
	ctx *types.Context, service string) error {
	return nil
}

func (d *driver) VolumesByServiceAfter(
	ctx types.Context, service string, result *types.VolumeMap) {
}

func (d *driver) VolumeInspectBefore(
	ctx *types.Context,
	service, volumeID string, attachments bool) error {
	return nil
}

func (d *driver) VolumeInspectAfter(
	ctx types.Context,
	result *types.Volume) {
}

func (d *driver) VolumeCreateBefore(
	ctx *types.Context,
	service string, request *types.VolumeCreateRequest) error {
	return nil
}

func (d *driver) VolumeCreateAfter(
	ctx types.Context,
	result *types.Volume) {
	volDir := path.Join(vfs.VolumesDirPath(d.config), result.ID)
	os.MkdirAll(volDir, 0755)
}

func (d *driver) VolumeCreateFromSnapshotBefore(
	ctx *types.Context,
	service, snapshotID string,
	request *types.VolumeCreateRequest) error {
	return nil
}

func (d *driver) VolumeCreateFromSnapshotAfter(
	ctx types.Context, result *types.Volume) {
	volDir := path.Join(vfs.VolumesDirPath(d.config), result.ID)
	os.MkdirAll(volDir, 0755)
}

func (d *driver) VolumeCopyBefore(
	ctx *types.Context,
	service, volumeID string, request *types.VolumeCopyRequest) error {
	return nil
}

func (d *driver) VolumeCopyAfter(
	ctx types.Context,
	result *types.Volume) {
	volDir := path.Join(vfs.VolumesDirPath(d.config), result.ID)
	os.MkdirAll(volDir, 0755)
}

func (d *driver) VolumeRemoveBefore(
	ctx *types.Context, service, volumeID string) error {
	return nil
}

func (d *driver) VolumeRemoveAfter(
	ctx types.Context, service, volumeID string) {
	volDir := path.Join(vfs.VolumesDirPath(d.config), volumeID)
	os.RemoveAll(volDir)
}

func (d *driver) VolumeSnapshotBefore(
	ctx *types.Context,
	service, volumeID string,
	request *types.VolumeSnapshotRequest) error {
	return nil
}

func (d *driver) VolumeSnapshotAfter(
	ctx types.Context, result *types.Snapshot) {
}

func (d *driver) VolumeAttachBefore(
	ctx *types.Context,
	service, volumeID string,
	request *types.VolumeAttachRequest) error {
	return nil
}

func (d *driver) VolumeAttachAfter(ctx types.Context, result *types.Volume) {
}

func (d *driver) VolumeDetachBefore(
	ctx *types.Context,
	service, volumeID string,
	request *types.VolumeDetachRequest) error {
	return nil
}

func (d *driver) VolumeDetachAfter(
	ctx types.Context, result *types.Volume) {
}

func (d *driver) SnapshotsBefore(ctx *types.Context) error {
	return nil
}

func (d *driver) SnapshotsAfter(
	ctx types.Context, result *types.ServiceSnapshotMap) {
}

func (d *driver) SnapshotsByServiceBefore(
	ctx *types.Context, service string) error {
	return nil
}

func (d *driver) SnapshotsByServiceAfter(
	ctx types.Context, service string, result *types.SnapshotMap) {
}

func (d *driver) SnapshotInspectBefore(
	ctx *types.Context,
	service, snapshotID string) error {
	return nil
}

func (d *driver) SnapshotInspectAfter(
	ctx types.Context, result *types.Volume) {
}

func (d *driver) SnapshotCopyBefore(
	ctx *types.Context,
	service, snapshotID, string,
	request *types.SnapshotCopyRequest) error {
	return nil
}

func (d *driver) SnapshotCopyAfter(
	ctx types.Context, result *types.Snapshot) {
}

func (d *driver) SnapshotRemoveBefore(
	ctx *types.Context, service, snapshotID string) error {
	return nil
}

func (d *driver) SnapshotRemoveAfter(ctx types.Context, snapshotID string) {
}
