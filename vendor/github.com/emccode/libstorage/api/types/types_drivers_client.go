package types

// NewClientDriver is a function that constructs a new ClientDriver.
type NewClientDriver func() ClientDriver

// ClientDriver is the client-side driver that is able to inspect
// methods before and after they are invoked in order to both prevent their
// execution as well as mutate the results.
type ClientDriver interface {
	Driver

	// InstanceInspectBefore may return an error, preventing the operation.
	InstanceInspectBefore(ctx *Context) error

	// InstanceInspectAfter provides an opportunity to inspect/mutate the
	// result.
	InstanceInspectAfter(ctx Context, result *Instance)

	// VolumesBefore may return an error, preventing the operation.
	VolumesBefore(ctx *Context) error

	// VolumesAfter provides an opportunity to inspect/mutate the result.
	VolumesAfter(ctx Context, result *ServiceVolumeMap)

	// VolumesByServiceBefore may return an error, preventing the operation.
	VolumesByServiceBefore(ctx *Context, service string) error

	// VolumesByServiceAfter provides an opportunity to inspect/mutate the
	// result.
	VolumesByServiceAfter(
		ctx Context, service string, result *VolumeMap)

	// VolumeInspectBefore may return an error, preventing the operation.
	VolumeInspectBefore(
		ctx *Context, service, volumeID string, attachments bool) error

	// VolumeInspectAfter provides an opportunity to inspect/mutate the result.
	VolumeInspectAfter(ctx Context, result *Volume)

	// VolumeCreateBefore may return an error, preventing the operation.
	VolumeCreateBefore(
		ctx *Context, service string,
		request *VolumeCreateRequest) error

	// VolumeCreateAfter provides an opportunity to inspect/mutate the result.
	VolumeCreateAfter(ctx Context, result *Volume)

	// VolumeCreateFromSnapshotBefore may return an error, preventing the
	// operation.
	VolumeCreateFromSnapshotBefore(
		ctx *Context,
		service, snapshotID string,
		request *VolumeCreateRequest) error

	// VolumeCreateFromSnapshotAfter provides an opportunity to inspect/mutate
	// the result.
	VolumeCreateFromSnapshotAfter(ctx Context, result *Volume)

	// VolumeCopyBefore may return an error, preventing the operation.
	VolumeCopyBefore(
		ctx *Context,
		service, volumeID string,
		request *VolumeCopyRequest) error

	// VolumeCopyAfter provides an opportunity to inspect/mutate the result.
	VolumeCopyAfter(ctx Context, result *Volume)

	// VolumeRemoveBefore may return an error, preventing the operation.
	VolumeRemoveBefore(
		ctx *Context,
		service, volumeID string) error

	// VolumeRemoveAfter provides an opportunity to inspect/mutate the result.
	VolumeRemoveAfter(ctx Context, service, volumeID string)

	// VolumeSnapshotBefore may return an error, preventing the operation.
	VolumeSnapshotBefore(
		ctx *Context,
		service, volumeID string,
		request *VolumeSnapshotRequest) error

	// VolumeSnapshotAfter provides an opportunity to inspect/mutate the result.
	VolumeSnapshotAfter(ctx Context, result *Snapshot)

	// VolumeAttachBefore may return an error, preventing the operation.
	VolumeAttachBefore(
		ctx *Context,
		service, volumeID string,
		request *VolumeAttachRequest) error

	// VolumeAttachAfter provides an opportunity to inspect/mutate the result.
	VolumeAttachAfter(ctx Context, result *Volume)

	// VolumeDetachBefore may return an error, preventing the operation.
	VolumeDetachBefore(
		ctx *Context,
		service, volumeID string,
		request *VolumeDetachRequest) error

	// VolumeDetachAfter provides an opportunity to inspect/mutate the result.
	VolumeDetachAfter(ctx Context, result *Volume)

	// SnapshotsBefore may return an error, preventing the operation.
	SnapshotsBefore(ctx *Context) error

	// SnapshotsAfter provides an opportunity to inspect/mutate the result.
	SnapshotsAfter(ctx Context, result *ServiceSnapshotMap)

	// SnapshotsByServiceBefore may return an error, preventing the operation.
	SnapshotsByServiceBefore(ctx *Context, service string) error

	// SnapshotsByServiceAfter provides an opportunity to inspect/mutate the
	// result.
	SnapshotsByServiceAfter(
		ctx Context, service string, result *SnapshotMap)

	// SnapshotInspectBefore may return an error, preventing the operation.
	SnapshotInspectBefore(
		ctx *Context,
		service, snapshotID string) error

	// SnapshotInspectAfter provides an opportunity to inspect/mutate the
	// result.
	SnapshotInspectAfter(ctx Context, result *Volume)

	// SnapshotCopyBefore may return an error, preventing the operation.
	SnapshotCopyBefore(
		ctx *Context,
		service, snapshotID, string,
		request *SnapshotCopyRequest) error

	// SnapshotCopyAfter provides an opportunity to inspect/mutate the result.
	SnapshotCopyAfter(ctx Context, result *Snapshot)

	// SnapshotRemoveBefore may return an error, preventing the operation.
	SnapshotRemoveBefore(ctx *Context, service, snapshotID string) error

	// SnapshotRemoveAfter provides an opportunity to inspect/mutate the result.
	SnapshotRemoveAfter(ctx Context, snapshotID string)
}
