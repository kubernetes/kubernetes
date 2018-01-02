package vfs

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"go.pedge.io/dlog"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/libopenstorage/openstorage/volume/drivers/common"
	"github.com/pborman/uuid"
	"github.com/portworx/kvdb"
)

const (
	// Name of the driver
	Name = "vfs"
	// Type of the driver
	Type = api.DriverType_DRIVER_TYPE_FILE
)

type driver struct {
	volume.IODriver
	volume.BlockDriver
	volume.SnapshotDriver
	volume.StoreEnumerator
	volume.StatsDriver
}

// Init Driver intialization.
func Init(params map[string]string) (volume.VolumeDriver, error) {
	return &driver{
		volume.IONotSupported,
		volume.BlockNotSupported,
		volume.SnapshotNotSupported,
		common.NewDefaultStoreEnumerator(Name, kvdb.Instance()),
		volume.StatsNotSupported,
	}, nil
}

func (d *driver) Name() string {
	return Name
}

func (d *driver) Type() api.DriverType {
	return Type
}

func (d *driver) Create(locator *api.VolumeLocator, source *api.Source, spec *api.VolumeSpec) (string, error) {
	volumeID := strings.TrimSuffix(uuid.New(), "\n")
	// Create a directory on the Local machine with this UUID.
	if err := os.MkdirAll(filepath.Join(volume.VolumeBase, string(volumeID)), 0744); err != nil {
		return "", err
	}
	v := common.NewVolume(
		volumeID,
		api.FSType_FS_TYPE_VFS,
		locator,
		source,
		spec,
	)
	v.DevicePath = filepath.Join(volume.VolumeBase, volumeID)
	if err := d.CreateVol(v); err != nil {
		return "", err
	}
	return v.Id, d.UpdateVol(v)
}

func (d *driver) Delete(volumeID string) error {
	if _, err := d.GetVol(volumeID); err != nil {
		return err
	}
	os.RemoveAll(filepath.Join(volume.VolumeBase, string(volumeID)))
	if err := d.DeleteVol(volumeID); err != nil {
		return err
	}
	return nil

}

func (d *driver) MountedAt(mountpath string) string {
	return ""
}

// Mount volume at specified path
// Errors ErrEnoEnt, ErrVolDetached may be returned.
func (d *driver) Mount(volumeID string, mountpath string) error {
	v, err := d.GetVol(volumeID)
	if err != nil {
		dlog.Println(err)
		return err
	}
	if len(v.AttachPath) > 0 && len(v.AttachPath) > 0 {
		return fmt.Errorf("Volume %q already mounted at %q", volumeID, v.AttachPath[0])
	}
	syscall.Unmount(mountpath, 0)
	if err := syscall.Mount(
		filepath.Join(volume.VolumeBase, string(volumeID)),
		mountpath,
		string(v.Spec.Format),
		syscall.MS_BIND, "",
	); err != nil {
		dlog.Printf("Cannot mount %s at %s because %+v",
			filepath.Join(volume.VolumeBase, string(volumeID)),
			mountpath,
			err,
		)
		return err
	}
	if v.AttachPath == nil {
		v.AttachPath = make([]string, 1)
	}
	v.AttachPath[0] = mountpath
	return d.UpdateVol(v)
}

// Unmount volume at specified path
// Errors ErrEnoEnt, ErrVolDetached may be returned.
func (d *driver) Unmount(volumeID string, mountpath string) error {
	v, err := d.GetVol(volumeID)
	if err != nil {
		return err
	}
	if len(v.AttachPath) == 0 || len(v.AttachPath[0]) == 0 {
		return fmt.Errorf("Device %v not mounted", volumeID)
	}
	if err := syscall.Unmount(v.AttachPath[0], 0); err != nil {
		return err
	}
	v.AttachPath = nil
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

func (d *driver) Status() [][2]string {
	return [][2]string{}
}

func (d *driver) Shutdown() {}
