package local

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/docker/docker/volume"
)

// VolumeDataPathName is the name of the directory where the volume data is stored.
// It uses a very distintive name to avoid colissions migrating data between
// Docker versions.
const (
	VolumeDataPathName = "_data"
	volumesPathName    = "volumes"
)

var oldVfsDir = filepath.Join("vfs", "dir")

func New(scope string) (*Root, error) {
	rootDirectory := filepath.Join(scope, volumesPathName)

	if err := os.MkdirAll(rootDirectory, 0700); err != nil {
		return nil, err
	}

	r := &Root{
		scope:   scope,
		path:    rootDirectory,
		volumes: make(map[string]*Volume),
	}

	dirs, err := ioutil.ReadDir(rootDirectory)
	if err != nil {
		return nil, err
	}

	for _, d := range dirs {
		name := filepath.Base(d.Name())
		r.volumes[name] = &Volume{
			driverName: r.Name(),
			name:       name,
			path:       r.DataPath(name),
		}
	}
	return r, nil
}

type Root struct {
	m       sync.Mutex
	scope   string
	path    string
	volumes map[string]*Volume
}

func (r *Root) DataPath(volumeName string) string {
	return filepath.Join(r.path, volumeName, VolumeDataPathName)
}

func (r *Root) Name() string {
	return "local"
}

func (r *Root) Create(name string) (volume.Volume, error) {
	r.m.Lock()
	defer r.m.Unlock()

	v, exists := r.volumes[name]
	if !exists {
		path := r.DataPath(name)
		if err := os.MkdirAll(path, 0755); err != nil {
			if os.IsExist(err) {
				return nil, fmt.Errorf("volume already exists under %s", filepath.Dir(path))
			}
			return nil, err
		}
		v = &Volume{
			driverName: r.Name(),
			name:       name,
			path:       path,
		}
		r.volumes[name] = v
	}
	v.use()
	return v, nil
}

func (r *Root) Remove(v volume.Volume) error {
	r.m.Lock()
	defer r.m.Unlock()
	lv, ok := v.(*Volume)
	if !ok {
		return errors.New("unknown volume type")
	}
	lv.release()
	if lv.usedCount == 0 {
		realPath, err := filepath.EvalSymlinks(lv.path)
		if err != nil {
			return err
		}
		if !r.scopedPath(realPath) {
			return fmt.Errorf("Unable to remove a directory of out the Docker root: %s", realPath)
		}

		if err := os.RemoveAll(realPath); err != nil {
			return err
		}

		delete(r.volumes, lv.name)
		return os.RemoveAll(filepath.Dir(lv.path))
	}
	return nil
}

// scopedPath verifies that the path where the volume is located
// is under Docker's root and the valid local paths.
func (r *Root) scopedPath(realPath string) bool {
	// Volumes path for Docker version >= 1.7
	if strings.HasPrefix(realPath, filepath.Join(r.scope, volumesPathName)) {
		return true
	}

	// Volumes path for Docker version < 1.7
	if strings.HasPrefix(realPath, filepath.Join(r.scope, oldVfsDir)) {
		return true
	}

	return false
}

type Volume struct {
	m         sync.Mutex
	usedCount int
	// unique name of the volume
	name string
	// path is the path on the host where the data lives
	path string
	// driverName is the name of the driver that created the volume.
	driverName string
}

func (v *Volume) Name() string {
	return v.name
}

func (v *Volume) DriverName() string {
	return v.driverName
}

func (v *Volume) Path() string {
	return v.path
}

func (v *Volume) Mount() (string, error) {
	return v.path, nil
}

func (v *Volume) Unmount() error {
	return nil
}

func (v *Volume) use() {
	v.m.Lock()
	v.usedCount++
	v.m.Unlock()
}

func (v *Volume) release() {
	v.m.Lock()
	v.usedCount--
	v.m.Unlock()
}
