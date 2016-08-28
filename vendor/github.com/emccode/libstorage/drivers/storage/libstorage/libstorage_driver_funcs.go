package libstorage

import (
	"github.com/akutz/goof"
	"github.com/emccode/libstorage/api/context"
	"github.com/emccode/libstorage/api/types"
	"github.com/emccode/libstorage/api/utils"
)

func (d *driver) Name() string {
	return Name
}

func (d *driver) API() types.APIClient {
	return &d.client
}

func (d *driver) XCLI() types.StorageExecutorCLI {
	return &d.client
}

func (d *driver) NextDeviceInfo(
	ctx types.Context) (*types.NextDeviceInfo, error) {

	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	si, err := d.getServiceInfo(serviceName)
	if err != nil {
		return nil, err
	}
	return si.Driver.NextDevice, nil
}

func (d *driver) Type(ctx types.Context) (types.StorageType, error) {

	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return "", goof.New("missing service name")
	}

	si, err := d.getServiceInfo(serviceName)
	if err != nil {
		return "", err
	}
	return si.Driver.Type, nil
}

func (d *driver) InstanceInspect(
	ctx types.Context,
	opts types.Store) (*types.Instance, error) {

	if d.isController() {
		return nil, utils.NewUnsupportedForClientTypeError(
			d.clientType, "InstanceInspect")
	}

	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	return d.client.InstanceInspect(ctx, serviceName)
}

func (d *driver) Volumes(
	ctx types.Context,
	opts *types.VolumesOpts) ([]*types.Volume, error) {

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	objMap, err := d.client.VolumesByService(ctx, serviceName, opts.Attachments)
	if err != nil {
		return nil, err
	}

	objs := []*types.Volume{}
	for _, o := range objMap {
		objs = append(objs, o)
	}

	return objs, nil
}

func (d *driver) VolumeInspect(
	ctx types.Context,
	volumeID string,
	opts *types.VolumeInspectOpts) (*types.Volume, error) {

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	return d.client.VolumeInspect(ctx, serviceName, volumeID, opts.Attachments)
}

func (d *driver) VolumeCreate(
	ctx types.Context,
	name string,
	opts *types.VolumeCreateOpts) (*types.Volume, error) {

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	req := &types.VolumeCreateRequest{
		Name:             name,
		AvailabilityZone: opts.AvailabilityZone,
		IOPS:             opts.IOPS,
		Size:             opts.Size,
		Type:             opts.Type,
		Opts:             opts.Opts.Map(),
	}

	return d.client.VolumeCreate(ctx, serviceName, req)
}

func (d *driver) VolumeCreateFromSnapshot(
	ctx types.Context,
	snapshotID, volumeName string,
	opts *types.VolumeCreateOpts) (*types.Volume, error) {

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	req := &types.VolumeCreateRequest{
		Name:             volumeName,
		AvailabilityZone: opts.AvailabilityZone,
		IOPS:             opts.IOPS,
		Size:             opts.Size,
		Type:             opts.Type,
		Opts:             opts.Opts.Map(),
	}

	return d.client.VolumeCreateFromSnapshot(ctx, serviceName, snapshotID, req)
}

func (d *driver) VolumeCopy(
	ctx types.Context,
	volumeID, volumeName string,
	opts types.Store) (*types.Volume, error) {

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	req := &types.VolumeCopyRequest{
		VolumeName: volumeName,
		Opts:       opts.Map(),
	}

	return d.client.VolumeCopy(ctx, serviceName, volumeID, req)
}

func (d *driver) VolumeSnapshot(
	ctx types.Context,
	volumeID, snapshotName string,
	opts types.Store) (*types.Snapshot, error) {

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	req := &types.VolumeSnapshotRequest{
		SnapshotName: snapshotName,
		Opts:         opts.Map(),
	}

	return d.client.VolumeSnapshot(ctx, serviceName, volumeID, req)
}

func (d *driver) VolumeRemove(
	ctx types.Context,
	volumeID string,
	opts types.Store) error {

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return goof.New("missing service name")
	}

	return d.client.VolumeRemove(ctx, serviceName, volumeID)
}

func (d *driver) VolumeAttach(
	ctx types.Context,
	volumeID string,
	opts *types.VolumeAttachOpts) (*types.Volume, string, error) {

	if d.isController() {
		return nil, "", utils.NewUnsupportedForClientTypeError(
			d.clientType, "VolumeAttach")
	}

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, "", goof.New("missing service name")
	}

	req := &types.VolumeAttachRequest{
		NextDeviceName: opts.NextDevice,
		Force:          opts.Force,
		Opts:           opts.Opts.Map(),
	}

	return d.client.VolumeAttach(ctx, serviceName, volumeID, req)
}

func (d *driver) VolumeDetach(
	ctx types.Context,
	volumeID string,
	opts *types.VolumeDetachOpts) (*types.Volume, error) {

	if d.isController() {
		return nil, utils.NewUnsupportedForClientTypeError(
			d.clientType, "VolumeDetach")
	}

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	req := &types.VolumeDetachRequest{
		Force: opts.Force,
		Opts:  opts.Opts.Map(),
	}

	return d.client.VolumeDetach(ctx, serviceName, volumeID, req)
}

func (d *driver) Snapshots(
	ctx types.Context,
	opts types.Store) ([]*types.Snapshot, error) {

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	objMap, err := d.client.SnapshotsByService(ctx, serviceName)

	if err != nil {
		return nil, err
	}

	objs := []*types.Snapshot{}
	for _, o := range objMap {
		objs = append(objs, o)
	}

	return objs, nil
}

func (d *driver) SnapshotInspect(
	ctx types.Context,
	snapshotID string,
	opts types.Store) (*types.Snapshot, error) {

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	return d.client.SnapshotInspect(ctx, serviceName, snapshotID)
}

func (d *driver) SnapshotCopy(
	ctx types.Context,
	snapshotID, snapshotName, destinationID string,
	opts types.Store) (*types.Snapshot, error) {

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return nil, goof.New("missing service name")
	}

	req := &types.SnapshotCopyRequest{
		SnapshotName:  snapshotName,
		DestinationID: destinationID,
		Opts:          opts.Map(),
	}

	return d.client.SnapshotCopy(ctx, serviceName, snapshotID, req)
}

func (d *driver) SnapshotRemove(
	ctx types.Context,
	snapshotID string,
	opts types.Store) error {

	ctx = d.requireCtx(ctx)
	serviceName, ok := context.ServiceName(ctx)
	if !ok {
		return goof.New("missing service name")
	}

	return d.client.SnapshotRemove(ctx, serviceName, snapshotID)
}

func (d *driver) assertProvidesAPIClient() types.ProvidesAPIClient {
	return d
}

func (d *driver) assertProvidesStorageExecutorCLI() types.ProvidesStorageExecutorCLI {
	return d
}
