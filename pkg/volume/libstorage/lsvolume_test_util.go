/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package libstorage

import (
	"fmt"
	"strconv"
	"time"

	"github.com/akutz/gofig"
	lstypes "github.com/emccode/libstorage/api/types"
)

type testClient struct {
	storageDriver     lstypes.StorageDriver
	integrationDriver lstypes.IntegrationDriver
}

func newTestClient() *testClient {
	driver := newTestStorageDriver()
	return &testClient{driver, driver}
}

// API returns the underlying libStorage API client.
func (c *testClient) API() lstypes.APIClient {
	return nil
}

// OS returns the client's OS driver instance.
func (c *testClient) OS() lstypes.OSDriver {
	return nil
}

// Storage returns the client's storage driver instance.
func (c *testClient) Storage() lstypes.StorageDriver {
	return c.storageDriver
}

// IntegrationDriver returns the client's integration driver instance.
func (c *testClient) Integration() lstypes.IntegrationDriver {
	return c.integrationDriver
}

// Executor returns the storage executor CLI.
func (c *testClient) Executor() lstypes.StorageExecutorCLI {
	return nil
}

type testStorageDriver struct {
	volumes []*lstypes.Volume
}

func newTestStorageDriver() *testStorageDriver {
	return new(testStorageDriver)
}

// Name returns the name of the driver
func (d *testStorageDriver) Name() string {
	return "test-storage-driver"
}

// Init initializes the driver.
func (d *testStorageDriver) Init(ctx lstypes.Context, config gofig.Config) error {
	return nil
}

// NextDeviceInfo returns the information about the driver's next available
// device workflow.
func (d *testStorageDriver) NextDeviceInfo(
	ctx lstypes.Context) (*lstypes.NextDeviceInfo, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// Type returns the type of storage the driver provides.
func (d *testStorageDriver) Type(
	ctx lstypes.Context) (lstypes.StorageType, error) {
	return "test-store", nil
}

// InstanceInspect returns an instance.
func (d *testStorageDriver) InstanceInspect(
	ctx lstypes.Context,
	opts lstypes.Store) (*lstypes.Instance, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// Volumes returns all volumes or a filtered list of volumes.
func (d *testStorageDriver) Volumes(
	ctx lstypes.Context,
	opts *lstypes.VolumesOpts) ([]*lstypes.Volume, error) {
	return d.volumes, nil
}

// VolumeInspect inspects a single volume.
func (d *testStorageDriver) VolumeInspect(
	ctx lstypes.Context,
	volumeID string,
	opts *lstypes.VolumeInspectOpts) (*lstypes.Volume, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// VolumeCreate creates a new volume.
func (d *testStorageDriver) VolumeCreate(
	ctx lstypes.Context,
	name string,
	opts *lstypes.VolumeCreateOpts) (*lstypes.Volume, error) {
	v := &lstypes.Volume{
		ID:   strconv.FormatInt(time.Now().UnixNano(), 10),
		Name: name,
		Size: *opts.IOPS,
	}
	d.volumes = append(d.volumes, v)
	return v, nil
}

// VolumeCreateFromSnapshot creates a new volume from an existing snapshot.
func (d *testStorageDriver) VolumeCreateFromSnapshot(
	ctx lstypes.Context,
	snapshotID,
	volumeName string,
	opts *lstypes.VolumeCreateOpts) (*lstypes.Volume, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// VolumeCopy copies an existing volume.
func (d *testStorageDriver) VolumeCopy(
	ctx lstypes.Context,
	volumeID,
	volumeName string,
	opts lstypes.Store) (*lstypes.Volume, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// VolumeSnapshot snapshots a volume.
func (d *testStorageDriver) VolumeSnapshot(
	ctx lstypes.Context,
	volumeID,
	snapshotName string,
	opts lstypes.Store) (*lstypes.Snapshot, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// VolumeRemove removes a volume.
func (d *testStorageDriver) VolumeRemove(
	ctx lstypes.Context,
	volumeID string,
	opts lstypes.Store) error {
	return fmt.Errorf("Unimplemented")
}

// VolumeAttach attaches a volume and provides a token clients can use
// to validate that device has appeared locally.
func (d *testStorageDriver) VolumeAttach(
	ctx lstypes.Context,
	volumeID string,
	opts *lstypes.VolumeAttachOpts) (*lstypes.Volume, string, error) {
	return nil, "", fmt.Errorf("Unimplemented")
}

// VolumeDetach detaches a volume.
func (d *testStorageDriver) VolumeDetach(
	ctx lstypes.Context,
	volumeID string,
	opts *lstypes.VolumeDetachOpts) (*lstypes.Volume, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// Snapshots returns all volumes or a filtered list of snapshots.
func (d *testStorageDriver) Snapshots(
	ctx lstypes.Context,
	opts lstypes.Store) ([]*lstypes.Snapshot, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// SnapshotInspect inspects a single snapshot.
func (d *testStorageDriver) SnapshotInspect(
	ctx lstypes.Context,
	snapshotID string,
	opts lstypes.Store) (*lstypes.Snapshot, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// SnapshotCopy copies an existing snapshot.
func (d *testStorageDriver) SnapshotCopy(
	ctx lstypes.Context,
	snapshotID,
	snapshotName,
	destinationID string,
	opts lstypes.Store) (*lstypes.Snapshot, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// SnapshotRemove removes a snapshot.
func (d *testStorageDriver) SnapshotRemove(
	ctx lstypes.Context,
	snapshotID string,
	opts lstypes.Store) error {
	return fmt.Errorf("Unimplemented")
}

// List a map that relates volume names to their mount points.
func (d *testStorageDriver) List(
	ctx lstypes.Context,
	opts lstypes.Store) ([]lstypes.VolumeMapping, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// Inspect returns a specific volume as identified by the provided
// volume name.
func (d *testStorageDriver) Inspect(
	ctx lstypes.Context,
	volumeName string,
	opts lstypes.Store) (lstypes.VolumeMapping, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// Mount will return a mount point path when specifying either a volumeName
// or volumeID.  If a overwriteFs boolean is specified it will overwrite
// the FS based on newFsType if it is detected that there is no FS present.
func (d *testStorageDriver) Mount(
	ctx lstypes.Context,
	volumeID, volumeName string,
	opts *lstypes.VolumeMountOpts) (string, *lstypes.Volume, error) {
	return "", nil, fmt.Errorf("Unimplemented")
}

// Unmount will unmount the specified volume by volumeName or volumeID.
func (d *testStorageDriver) Unmount(
	ctx lstypes.Context,
	volumeID, volumeName string,
	opts lstypes.Store) error {
	return fmt.Errorf("Unimplemented")
}

// Path will return the mounted path of the volumeName or volumeID.
func (d *testStorageDriver) Path(
	ctx lstypes.Context,
	volumeID, volumeName string,
	opts lstypes.Store) (string, error) {
	return "", fmt.Errorf("Unimplemented")
}

// Create will create a new volume with the volumeName and opts.
func (d *testStorageDriver) Create(
	ctx lstypes.Context,
	volumeName string,
	opts *lstypes.VolumeCreateOpts) (*lstypes.Volume, error) {
	return nil, fmt.Errorf("Unimplemented")
}

// Remove will remove a volume of volumeName.
func (d *testStorageDriver) Remove(
	ctx lstypes.Context,
	volumeName string,
	opts lstypes.Store) error {
	return fmt.Errorf("Unimplemented")
}

// Attach will attach a volume based on volumeName to the instance of
// instanceID.
func (d *testStorageDriver) Attach(
	ctx lstypes.Context,
	volumeName string,
	opts *lstypes.VolumeAttachOpts) (string, error) {
	return "", fmt.Errorf("Unimplemented")
}

// Detach will detach a volume based on volumeName to the instance of
// instanceID.
func (d *testStorageDriver) Detach(
	ctx lstypes.Context,
	volumeName string,
	opts *lstypes.VolumeDetachOpts) error {
	return fmt.Errorf("Unimplemented")
}
