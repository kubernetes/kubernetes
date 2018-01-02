package nfs

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"syscall"

	"go.pedge.io/dlog"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/config"
	"github.com/libopenstorage/openstorage/pkg/mount"
	"github.com/libopenstorage/openstorage/pkg/seed"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/libopenstorage/openstorage/volume/drivers/common"
	"github.com/portworx/kvdb"
)

const (
	Name         = "nfs"
	Type         = api.DriverType_DRIVER_TYPE_FILE
	NfsDBKey     = "OpenStorageNFSKey"
	nfsMountPath = "/var/lib/openstorage/nfs/"
	nfsBlockFile = ".blockdevice"
)

// Implements the open storage volume interface.
type driver struct {
	volume.IODriver
	volume.StoreEnumerator
	volume.StatsDriver
	nfsServer string
	nfsPath   string
	mounter   mount.Manager
}

func Init(params map[string]string) (volume.VolumeDriver, error) {
	path, ok := params["path"]
	if !ok {
		return nil, errors.New("No NFS path provided")
	}
	server, ok := params["server"]
	if !ok {
		dlog.Printf("No NFS server provided, will attempt to bind mount %s", path)
	} else {
		dlog.Printf("NFS driver initializing with %s:%s ", server, path)
	}
	// Create a mount manager for this NFS server. Blank sever is OK.
	mounter, err := mount.New(mount.NFSMount, nil, []string{server}, nil, []string{})
	if err != nil {
		dlog.Warnf("Failed to create mount manager for server: %v (%v)", server, err)
		return nil, err
	}
	inst := &driver{
		IODriver:        volume.IONotSupported,
		StoreEnumerator: common.NewDefaultStoreEnumerator(Name, kvdb.Instance()),
		StatsDriver:     volume.StatsNotSupported,
		nfsServer:       server,
		nfsPath:         path,
		mounter:         mounter,
	}
	if err := os.MkdirAll(nfsMountPath, 0744); err != nil {
		return nil, err
	}
	src := inst.nfsPath
	if server != "" {
		src = ":" + inst.nfsPath
	}
	// If src is already mounted at dest, leave it be.
	mountExists, err := mounter.Exists(src, nfsMountPath)
	if !mountExists {
		// Mount the nfs server locally on a unique path.
		syscall.Unmount(nfsMountPath, 0)
		if server != "" {
			err = syscall.Mount(
				src,
				nfsMountPath,
				"nfs",
				0,
				"nolock,addr="+inst.nfsServer,
			)
		} else {
			err = syscall.Mount(src, nfsMountPath, "", syscall.MS_BIND, "")
		}
		if err != nil {
			dlog.Printf("Unable to mount %s:%s at %s (%+v)",
				inst.nfsServer, inst.nfsPath, nfsMountPath, err)
			return nil, err
		}
	}
	volumeInfo, err := inst.StoreEnumerator.Enumerate(&api.VolumeLocator{}, nil)
	if err == nil {
		for _, info := range volumeInfo {
			if info.Status == api.VolumeStatus_VOLUME_STATUS_NONE {
				info.Status = api.VolumeStatus_VOLUME_STATUS_UP
				inst.UpdateVol(info)
			}
		}
	}

	dlog.Println("NFS initialized and driver mounted at: ", nfsMountPath)
	return inst, nil
}

func (d *driver) Name() string {
	return Name
}

func (d *driver) Type() api.DriverType {
	return Type
}

// Status diagnostic information
func (d *driver) Status() [][2]string {
	return [][2]string{}
}

//
// These functions below implement the volume driver interface.
//

func (d *driver) Create(
	locator *api.VolumeLocator,
	source *api.Source,
	spec *api.VolumeSpec) (string, error) {

	volumeID := locator.Name
	if _, err := d.GetVol(volumeID); err == nil {
		return "", errors.New("Volume with that name already exists")
	}

	// Create a directory on the NFS server with this UUID.
	volPath := path.Join(nfsMountPath, volumeID)
	err := os.MkdirAll(volPath, 0744)
	if err != nil {
		dlog.Println(err)
		return "", err
	}
	if source != nil {
		if len(source.Seed) != 0 {
			seed, err := seed.New(source.Seed, spec.VolumeLabels)
			if err != nil {
				dlog.Warnf("Failed to initailize seed from %q : %v",
					source.Seed, err)
				return "", err
			}
			err = seed.Load(path.Join(volPath, config.DataDir))
			if err != nil {
				dlog.Warnf("Failed to  seed from %q to %q: %v",
					source.Seed, nfsMountPath, err)
				return "", err
			}
		}
	}

	f, err := os.Create(path.Join(nfsMountPath, volumeID+nfsBlockFile))
	if err != nil {
		dlog.Println(err)
		return "", err
	}
	defer f.Close()

	if err := f.Truncate(int64(spec.Size)); err != nil {
		dlog.Println(err)
		return "", err
	}

	v := common.NewVolume(
		volumeID,
		api.FSType_FS_TYPE_NFS,
		locator,
		source,
		spec,
	)
	v.DevicePath = path.Join(nfsMountPath, volumeID+nfsBlockFile)

	if err := d.CreateVol(v); err != nil {
		return "", err
	}
	return v.Id, err
}

func (d *driver) Delete(volumeID string) error {
	v, err := d.GetVol(volumeID)
	if err != nil {
		dlog.Println(err)
		return err
	}

	// Delete the simulated block volume
	os.Remove(v.DevicePath)

	// Delete the directory on the nfs server.
	os.RemoveAll(path.Join(nfsMountPath, volumeID))

	err = d.DeleteVol(volumeID)
	if err != nil {
		dlog.Println(err)
		return err
	}

	return nil
}

func (d *driver) MountedAt(mountpath string) string {
	return ""
}

func (d *driver) Mount(volumeID string, mountpath string) error {
	v, err := d.GetVol(volumeID)
	if err != nil {
		dlog.Println(err)
		return err
	}

	srcPath := path.Join(":", d.nfsPath, volumeID)
	mountExists, err := d.mounter.Exists(srcPath, mountpath)
	if !mountExists {
		d.mounter.Unmount(path.Join(nfsMountPath, volumeID), mountpath, syscall.MNT_DETACH, 0)
		if err := d.mounter.Mount(
			0, path.Join(nfsMountPath, volumeID),
			mountpath,
			string(v.Spec.Format),
			syscall.MS_BIND,
			"",
			0,
		); err != nil {
			dlog.Printf("Cannot mount %s at %s because %+v",
				path.Join(nfsMountPath, volumeID), mountpath, err)
			return err
		}
	}
	if v.AttachPath == nil {
		v.AttachPath = make([]string, 0)
	}
	v.AttachPath = append(v.AttachPath, mountpath)
	return d.UpdateVol(v)
}

func (d *driver) Unmount(volumeID string, mountpath string) error {
	v, err := d.GetVol(volumeID)
	if err != nil {
		return err
	}
	if len(v.AttachPath) == 0 {
		return fmt.Errorf("Device %v not mounted", volumeID)
	}
	err = d.mounter.Unmount(path.Join(nfsMountPath, volumeID), mountpath, syscall.MNT_DETACH, 0)
	if err != nil {
		return err
	}
	v.AttachPath = d.mounter.Mounts(path.Join(nfsMountPath, volumeID))
	return d.UpdateVol(v)
}

func (d *driver) Snapshot(volumeID string, readonly bool, locator *api.VolumeLocator) (string, error) {
	volIDs := []string{volumeID}
	vols, err := d.Inspect(volIDs)
	if err != nil {
		return "", nil
	}
	source := &api.Source{Parent: volumeID}
	newVolumeID, err := d.Create(locator, source, vols[0].Spec)
	if err != nil {
		return "", nil
	}

	// NFS does not support snapshots, so just copy the files.
	if err := copyDir(nfsMountPath+volumeID, nfsMountPath+newVolumeID); err != nil {
		d.Delete(newVolumeID)
		return "", nil
	}
	return newVolumeID, nil
}

func (d *driver) Restore(volumeID string, snapID string) error {
	if _, err := d.Inspect([]string{volumeID, snapID}); err != nil {
		return err
	}

	// NFS does not support restore, so just copy the files.
	if err := copyDir(nfsMountPath+snapID, nfsMountPath+volumeID); err != nil {
		return err
	}
	return nil
}

func (d *driver) Attach(volumeID string, attachOptions map[string]string) (string, error) {
	return path.Join(nfsMountPath, volumeID+nfsBlockFile), nil
}

func (d *driver) Detach(volumeID string, unmountBeforeDetach bool) error {
	return nil
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

func (d *driver) Shutdown() {
	dlog.Printf("%s Shutting down", Name)
	syscall.Unmount(nfsMountPath, 0)
}

func copyFile(source string, dest string) (err error) {
	sourcefile, err := os.Open(source)
	if err != nil {
		return err
	}

	defer sourcefile.Close()

	destfile, err := os.Create(dest)
	if err != nil {
		return err
	}

	defer destfile.Close()

	_, err = io.Copy(destfile, sourcefile)
	if err == nil {
		sourceinfo, err := os.Stat(source)
		if err != nil {
			err = os.Chmod(dest, sourceinfo.Mode())
		}

	}

	return
}

func copyDir(source string, dest string) (err error) {
	// get properties of source dir
	sourceinfo, err := os.Stat(source)
	if err != nil {
		return err
	}

	// create dest dir

	err = os.MkdirAll(dest, sourceinfo.Mode())
	if err != nil {
		return err
	}

	directory, _ := os.Open(source)

	objects, err := directory.Readdir(-1)

	for _, obj := range objects {

		sourcefilepointer := source + "/" + obj.Name()

		destinationfilepointer := dest + "/" + obj.Name()

		if obj.IsDir() {
			// create sub-directories - recursively
			err = copyDir(sourcefilepointer, destinationfilepointer)
			if err != nil {
				fmt.Println(err)
			}
		} else {
			// perform copy
			err = copyFile(sourcefilepointer, destinationfilepointer)
			if err != nil {
				fmt.Println(err)
			}
		}

	}
	return
}
