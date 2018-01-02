package layer0

import (
	"fmt"
	"os"
	"path"
	"strings"
	"sync"
	"sync/atomic"

	"go.pedge.io/dlog"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/daemon/graphdriver/overlay"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/parsers"
	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/graph"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/libopenstorage/openstorage/volume/drivers"
)

// Layer0 implemenation piggy backs on existing overlay graphdriver implementation
// to provide persistent storage for the uppermost/writeable layer in the
// container rootfs. The persistent storage is derived from one of the OSD volume drivers.
// To use this as the graphdriver in Docker with aws as the backend volume provider:
//
// DOCKER_STORAGE_OPTIONS= -s layer0 --storage-opt layer0.volume_driver=aws

// Layer0Vol represents the volume
type Layer0Vol struct {
	// id self referential ID
	id string
	// parent image string
	parent string
	// path where the external volume is mounted.
	path string
	// volumeID mapping to this external volume
	volumeID string
	// ref keeps track of mount and unmounts.
	ref int32
}

// Layer0 implements the graphdriver interface
type Layer0 struct {
	sync.Mutex
	// Driver is an implementation of GraphDriver. Only select methods are overridden
	graphdriver.Driver
	// home base string
	home string
	// volumes maintains a map of currently mounted volumes.
	volumes map[string]*Layer0Vol
	// volDriver is the volume driver used for the writeable layer.
	volDriver volume.VolumeDriver
}

// Layer0Graphdriver options. This should be passed in as a st
const (
	// Name of the driver
	Name = "layer0"
	// Type of the driver
	Type = api.DriverType_DRIVER_TYPE_GRAPH
	// Layer0VolumeDriver constant
	Layer0VolumeDriver = "layer0.volume_driver"
)

func init() {
	graph.Register(Name, Init)
}

// Init initializes the driver
func Init(home string, options []string, uidMaps, gidMaps []idtools.IDMap) (graphdriver.Driver, error) {
	var volumeDriver string
	for _, option := range options {
		key, val, err := parsers.ParseKeyValueOpt(option)
		if err != nil {
			return nil, err
		}
		switch key {
		case Layer0VolumeDriver:
			volumeDriver = val
		default:
			return nil, fmt.Errorf("Unknown option %s\n", key)
		}
	}
	dlog.Infof("Layer0 volume driver: %v", volumeDriver)
	volDriver, err := volumedrivers.Get(volumeDriver)
	if err != nil {
		return nil, err
	}
	ov, err := overlay.Init(home, options, uidMaps, gidMaps)
	if err != nil {
		volDriver.Shutdown()
		return nil, err
	}
	d := &Layer0{
		Driver:    ov,
		home:      home,
		volumes:   make(map[string]*Layer0Vol),
		volDriver: volDriver,
	}

	return d, nil
}

func (l *Layer0) isLayer0Parent(id string) (string, bool) {
	// This relies on an <instance_id>-init volume being created for
	// every new container.
	if strings.HasSuffix(id, "-init") {
		return strings.TrimSuffix(id, "-init"), true
	}
	return "", false
}

func (l *Layer0) isLayer0(id string) bool {
	if strings.HasSuffix(id, "-init") {
		baseID := strings.TrimSuffix(id, "-init")
		if _, ok := l.volumes[baseID]; !ok {
			l.volumes[baseID] = &Layer0Vol{id: baseID}
		}
		return false
	}
	_, ok := l.volumes[id]
	return ok
}

func (l *Layer0) loID(id string) string {
	return id + "-vol"
}

func (l *Layer0) realID(id string) string {
	if l.isLayer0(id) {
		return path.Join(l.loID(id), id)
	}
	return id
}

func (l *Layer0) create(id, parent string) (string, *Layer0Vol, error) {
	l.Lock()
	defer l.Unlock()

	// If this is the parent of the Layer0, add an entry for it.
	baseID, l0 := l.isLayer0Parent(id)
	if l0 {
		l.volumes[baseID] = &Layer0Vol{id: baseID, parent: parent}
		return id, nil, nil
	}

	// Don't do anything if this is not layer 0
	if !l.isLayer0(id) {
		return id, nil, nil
	}

	vol, ok := l.volumes[id]
	if !ok {
		dlog.Warnf("Failed to find layer0 volume for id %v", id)
		return id, nil, nil
	}

	// Query volume for Layer 0
	vols, err := l.volDriver.Enumerate(&api.VolumeLocator{Name: vol.parent}, nil)

	// If we don't find a volume configured for this image,
	// then don't track layer0
	if err != nil || vols == nil {
		dlog.Infof("Failed to find configured volume for id %v", vol.parent)
		delete(l.volumes, id)
		return id, nil, nil
	}

	// Find a volume that is available.
	index := -1
	for i, v := range vols {
		if len(v.AttachPath) == 0 {
			index = i
			break
		}
	}
	if index == -1 {
		dlog.Infof("Failed to find free volume for id %v", vol.parent)
		delete(l.volumes, id)
		return id, nil, nil
	}

	mountPath := path.Join(l.home, l.loID(id))
	os.MkdirAll(mountPath, 0755)

	// If this is a block driver, first attach the volume.
	if l.volDriver.Type() == api.DriverType_DRIVER_TYPE_BLOCK {
		_, err := l.volDriver.Attach(vols[index].Id, nil)
		if err != nil {
			dlog.Errorf("Failed to attach volume %v", vols[index].Id)
			delete(l.volumes, id)
			return id, nil, nil
		}
	}
	err = l.volDriver.Mount(vols[index].Id, mountPath)
	if err != nil {
		dlog.Errorf("Failed to mount volume %v at path %v",
			vols[index].Id, mountPath)
		delete(l.volumes, id)
		return id, nil, nil
	}
	vol.path = mountPath
	vol.volumeID = vols[index].Id
	vol.ref = 1

	return l.realID(id), vol, nil
}

// Create creates a new and empty filesystem layer
func (l *Layer0) Create(id string, parent string, mountLabel string, storageOpts map[string]string) error {
	id, vol, err := l.create(id, parent)
	if err != nil {
		return err
	}
	err = l.Driver.Create(id, parent, mountLabel, storageOpts)
	if err != nil || vol == nil {
		return err
	}
	// This is layer0. Restore saved upper dir, if one exists.
	savedUpper := path.Join(vol.path, "upper")
	if _, err := os.Stat(savedUpper); err != nil {
		// It's not an error if didn't have a saved upper
		return nil
	}
	// We found a saved upper, restore to newly created upper.
	upperDir := path.Join(path.Join(l.home, id), "upper")
	os.RemoveAll(upperDir)
	return os.Rename(savedUpper, upperDir)
}

// Remove removes a layer based on its id
func (l *Layer0) Remove(id string) error {
	if !l.isLayer0(id) {
		return l.Driver.Remove(l.realID(id))
	}
	l.Lock()
	defer l.Unlock()
	var err error
	v, ok := l.volumes[id]

	if ok {
		atomic.AddInt32(&v.ref, -1)
		if v.ref == 0 {
			// Save the upper dir and blow away the rest.
			upperDir := path.Join(path.Join(l.home, l.realID(id)), "upper")
			err := os.Rename(upperDir, path.Join(v.path, "upper"))
			if err != nil {
				dlog.Warnf("Failed in rename(%v): %v", id, err)
			}
			l.Driver.Remove(l.realID(id))
			err = l.volDriver.Unmount(v.volumeID, v.path)
			if l.volDriver.Type() == api.DriverType_DRIVER_TYPE_BLOCK {
				_ = l.volDriver.Detach(v.volumeID, false)
			}
			err = os.RemoveAll(v.path)
			delete(l.volumes, v.id)
		}
	} else {
		dlog.Warnf("Failed to find layer0 vol for id %v", id)
	}
	return err
}

// Get returns the mountpoint for the layered filesystem
func (l *Layer0) Get(id string, mountLabel string) (string, error) {
	id = l.realID(id)
	return l.Driver.Get(id, mountLabel)
}

// Put releases the system resources for the specified id
func (l *Layer0) Put(id string) error {
	id = l.realID(id)
	return l.Driver.Put(id)
}

// ApplyDiff extracts the changeset between the specified layer and its parent
func (l *Layer0) ApplyDiff(id string, parent string, diff archive.Reader) (size int64, err error) {
	id = l.realID(id)
	return l.Driver.ApplyDiff(id, parent, diff)
}

// Exists checks if leyr exists
func (l *Layer0) Exists(id string) bool {
	id = l.realID(id)
	return l.Driver.Exists(id)
}

// GetMetadata returns key-value pairs
func (l *Layer0) GetMetadata(id string) (map[string]string, error) {
	id = l.realID(id)
	return l.Driver.GetMetadata(id)
}
