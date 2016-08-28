package registry

import (
	"github.com/emccode/libstorage/api/types"
)

type sdm struct {
	types.StorageDriver
	types.Context
}

// NewStorageDriverManager returns a new storage driver manager.
func NewStorageDriverManager(
	d types.StorageDriver) types.StorageDriver {
	return &sdm{StorageDriver: d}
}

func (d *sdm) API() types.APIClient {
	if sd, ok := d.StorageDriver.(types.ProvidesAPIClient); ok {
		return sd.API()
	}
	return nil
}

func (d *sdm) XCLI() types.StorageExecutorCLI {
	if sd, ok := d.StorageDriver.(types.ProvidesStorageExecutorCLI); ok {
		return sd.XCLI()
	}
	return nil
}

func (d *sdm) NextDeviceInfo(
	ctx types.Context) (*types.NextDeviceInfo, error) {

	return d.StorageDriver.NextDeviceInfo(ctx.Join(d.Context))
}

func (d *sdm) Type(
	ctx types.Context) (types.StorageType, error) {

	return d.StorageDriver.Type(ctx.Join(d.Context))
}

func (d *sdm) InstanceInspect(
	ctx types.Context,
	opts types.Store) (*types.Instance, error) {

	return d.StorageDriver.InstanceInspect(ctx.Join(d.Context), opts)
}

func (d *sdm) Volumes(
	ctx types.Context,
	opts *types.VolumesOpts) ([]*types.Volume, error) {

	return d.StorageDriver.Volumes(ctx.Join(d.Context), opts)
}

func (d *sdm) VolumeInspect(
	ctx types.Context,
	volumeID string,
	opts *types.VolumeInspectOpts) (*types.Volume, error) {

	return d.StorageDriver.VolumeInspect(ctx.Join(d.Context), volumeID, opts)
}

func (d *sdm) VolumeCreate(
	ctx types.Context,
	name string,
	opts *types.VolumeCreateOpts) (*types.Volume, error) {

	return d.StorageDriver.VolumeCreate(ctx.Join(d.Context), name, opts)
}

func (d *sdm) VolumeCreateFromSnapshot(
	ctx types.Context,
	snapshotID,
	volumeName string,
	opts *types.VolumeCreateOpts) (*types.Volume, error) {

	return d.StorageDriver.VolumeCreateFromSnapshot(
		ctx.Join(d.Context), snapshotID, volumeName, opts)
}

func (d *sdm) VolumeCopy(
	ctx types.Context,
	volumeID,
	volumeName string,
	opts types.Store) (*types.Volume, error) {

	return d.StorageDriver.VolumeCopy(
		ctx.Join(d.Context), volumeID, volumeName, opts)
}

func (d *sdm) VolumeSnapshot(
	ctx types.Context,
	volumeID,
	snapshotName string,
	opts types.Store) (*types.Snapshot, error) {

	return d.StorageDriver.VolumeSnapshot(
		ctx.Join(d.Context), volumeID, snapshotName, opts)
}

func (d *sdm) VolumeRemove(
	ctx types.Context,
	volumeID string,
	opts types.Store) error {

	return d.StorageDriver.VolumeRemove(
		ctx.Join(d.Context), volumeID, opts)
}

func (d *sdm) VolumeAttach(
	ctx types.Context,
	volumeID string,
	opts *types.VolumeAttachOpts) (*types.Volume, string, error) {

	return d.StorageDriver.VolumeAttach(
		ctx.Join(d.Context), volumeID, opts)
}

func (d *sdm) VolumeDetach(
	ctx types.Context,
	volumeID string,
	opts *types.VolumeDetachOpts) (*types.Volume, error) {

	return d.StorageDriver.VolumeDetach(
		ctx.Join(d.Context), volumeID, opts)
}

func (d *sdm) Snapshots(
	ctx types.Context,
	opts types.Store) ([]*types.Snapshot, error) {

	return d.StorageDriver.Snapshots(ctx.Join(d.Context), opts)
}

func (d *sdm) SnapshotInspect(
	ctx types.Context,
	snapshotID string,
	opts types.Store) (*types.Snapshot, error) {

	return d.StorageDriver.SnapshotInspect(
		ctx.Join(d.Context), snapshotID, opts)
}

func (d *sdm) SnapshotCopy(
	ctx types.Context,
	snapshotID,
	snapshotName,
	destinationID string,
	opts types.Store) (*types.Snapshot, error) {

	return d.StorageDriver.SnapshotCopy(
		ctx.Join(d.Context), snapshotID, snapshotName, destinationID, opts)
}

func (d *sdm) SnapshotRemove(
	ctx types.Context,
	snapshotID string,
	opts types.Store) error {

	return d.StorageDriver.SnapshotRemove(ctx.Join(d.Context), snapshotID, opts)
}
