// +build linux,have_btrfs

package btrfs

import (
	"fmt"
	"path/filepath"
	"syscall"

	"go.pedge.io/proto/time"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/daemon/graphdriver/btrfs"
	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/pkg/chaos"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/libopenstorage/openstorage/volume/drivers/common"
	"github.com/pborman/uuid"
	"github.com/portworx/kvdb"
)

const (
	Name      = "btrfs"
	Type      = api.DriverType_DRIVER_TYPE_FILE
	RootParam = "home"
	Volumes   = "volumes"
)

var (
	koStrayCreate chaos.ID
	koStrayDelete chaos.ID
)

type driver struct {
	volume.StoreEnumerator
	volume.IODriver
	volume.BlockDriver
	btrfs graphdriver.Driver
	root  string
}

func Init(params map[string]string) (volume.VolumeDriver, error) {
	root, ok := params[RootParam]
	if !ok {
		return nil, fmt.Errorf("Root directory should be specified with key %q", RootParam)
	}
	home := filepath.Join(root, "volumes")
	d, err := btrfs.Init(home, nil, nil, nil)
	if err != nil {
		return nil, err
	}
	return &driver{
		common.NewDefaultStoreEnumerator(Name, kvdb.Instance()),
		common.IONotSupported,
		common.BlockNotSupported,
		d,
		root,
	}, nil
}

func (d *driver) Name() string {
	return Name
}

func (d *driver) Status() [][2]string {
	return d.btrfs.Status()
}

func (d *driver) Type() api.DriverType {
	return Type
}

// Create a new subvolume. The volume spec is not taken into account.
func (d *driver) Create(
	locator *api.VolumeLocator,
	source *api.Source,
	spec *api.VolumeSpec,
) (string, error) {
	if spec.Format != api.FSType_FS_TYPE_BTRFS && spec.Format != api.FSType_FS_TYPE_NONE {
		return "", fmt.Errorf("Filesystem format (%v) must be %v", spec.Format.SimpleString(), api.FSType_FS_TYPE_BTRFS.SimpleString())
	}
	volume := common.NewVolume(
		uuid.New(),
		api.FSType_FS_TYPE_BTRFS,
		locator,
		source,
		spec,
	)
	if err := d.CreateVol(volume); err != nil {
		return "", err
	}
	if err := d.btrfs.Create(volume.Id, "", "", nil); err != nil {
		return "", err
	}
	devicePath, err := d.btrfs.Get(volume.Id, "")
	if err != nil {
		return volume.Id, err
	}
	volume.DevicePath = devicePath
	return volume.Id, d.UpdateVol(volume)
}

func (d *driver) Delete(volumeID string) error {
	if err := d.DeleteVol(volumeID); err != nil {
		return err
	}
	chaos.Now(koStrayDelete)
	return d.btrfs.Remove(volumeID)
}

func (d *driver) Mount(volumeID string, mountpath string) error {
	v, err := d.GetVol(volumeID)
	if err != nil {
		return err
	}
	if err := syscall.Mount(v.DevicePath, mountpath, v.Format.SimpleString(), syscall.MS_BIND, ""); err != nil {
		return fmt.Errorf("Failed to mount %v at %v: %v", v.DevicePath, mountpath, err)
	}
	v.AttachPath = mountpath
	return d.UpdateVol(v)
}

func (d *driver) Unmount(volumeID string, mountpath string) error {
	v, err := d.GetVol(volumeID)
	if err != nil {
		return err
	}
	if v.AttachPath == "" {
		return fmt.Errorf("Device %v not mounted", volumeID)
	}
	if err := syscall.Unmount(v.AttachPath, 0); err != nil {
		return err
	}
	v.AttachPath = ""
	return d.UpdateVol(v)
}

func (d *driver) Set(volumeID string, locator *api.VolumeLocator, spec *api.VolumeSpec) error {
	if spec != nil {
		return volume.ErrNotSupported
	}
	v, err := d.GetVol(volumeID)
	if err != nil {
		return err
	}
	if locator != nil {
		v.Locator = locator
	}
	return d.UpdateVol(v)
}

// Snapshot create new subvolume from volume
func (d *driver) Snapshot(volumeID string, readonly bool, locator *api.VolumeLocator) (string, error) {
	vols, err := d.Inspect([]string{volumeID})
	if err != nil {
		return "", err
	}
	if len(vols) != 1 {
		return "", fmt.Errorf("Failed to inspect %v len %v", volumeID, len(vols))
	}
	snapID := uuid.New()
	vols[0].Id = snapID
	vols[0].Source = &api.Source{Parent: volumeID}
	vols[0].Locator = locator
	vols[0].Ctime = prototime.Now()

	if err := d.CreateVol(vols[0]); err != nil {
		return "", err
	}
	chaos.Now(koStrayCreate)
	err = d.btrfs.Create(snapID, volumeID, "", nil)
	if err != nil {
		return "", err
	}
	return vols[0].Id, nil
}

func (d *driver) Stats(volumeID string) (*api.Stats, error) {
	return nil, nil
}

func (d *driver) Alerts(volumeID string) (*api.Alerts, error) {
	return nil, nil
}

func (d *driver) Shutdown() {}
