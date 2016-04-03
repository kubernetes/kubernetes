// +build !windows

package daemon

import (
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/docker/docker/daemon/execdriver"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/volume"
	"github.com/docker/docker/volume/local"
)

// copyOwnership copies the permissions and uid:gid of the source file
// into the destination file
func copyOwnership(source, destination string) error {
	stat, err := system.Stat(source)
	if err != nil {
		return err
	}

	if err := os.Chown(destination, int(stat.Uid()), int(stat.Gid())); err != nil {
		return err
	}

	return os.Chmod(destination, os.FileMode(stat.Mode()))
}

func (container *Container) setupMounts() ([]execdriver.Mount, error) {
	var mounts []execdriver.Mount
	for _, m := range container.MountPoints {
		path, err := m.Setup()
		if err != nil {
			return nil, err
		}

		mounts = append(mounts, execdriver.Mount{
			Source:      path,
			Destination: m.Destination,
			Writable:    m.RW,
		})
	}

	mounts = sortMounts(mounts)
	return append(mounts, container.networkMounts()...), nil
}

func sortMounts(m []execdriver.Mount) []execdriver.Mount {
	sort.Sort(mounts(m))
	return m
}

type mounts []execdriver.Mount

func (m mounts) Len() int {
	return len(m)
}

func (m mounts) Less(i, j int) bool {
	return m.parts(i) < m.parts(j)
}

func (m mounts) Swap(i, j int) {
	m[i], m[j] = m[j], m[i]
}

func (m mounts) parts(i int) int {
	return len(strings.Split(filepath.Clean(m[i].Destination), string(os.PathSeparator)))
}

// migrateVolume links the contents of a volume created pre Docker 1.7
// into the location expected by the local driver.
// It creates a symlink from DOCKER_ROOT/vfs/dir/VOLUME_ID to DOCKER_ROOT/volumes/VOLUME_ID/_container_data.
// It preserves the volume json configuration generated pre Docker 1.7 to be able to
// downgrade from Docker 1.7 to Docker 1.6 without losing volume compatibility.
func migrateVolume(id, vfs string) error {
	l, err := getVolumeDriver(volume.DefaultDriverName)
	if err != nil {
		return err
	}

	newDataPath := l.(*local.Root).DataPath(id)
	fi, err := os.Stat(newDataPath)
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	if fi != nil && fi.IsDir() {
		return nil
	}

	return os.Symlink(vfs, newDataPath)
}

// validVolumeLayout checks whether the volume directory layout
// is valid to work with Docker post 1.7 or not.
func validVolumeLayout(files []os.FileInfo) bool {
	if len(files) == 1 && files[0].Name() == local.VolumeDataPathName && files[0].IsDir() {
		return true
	}

	if len(files) != 2 {
		return false
	}

	for _, f := range files {
		if f.Name() == "config.json" ||
			(f.Name() == local.VolumeDataPathName && f.Mode()&os.ModeSymlink == os.ModeSymlink) {
			// Old volume configuration, we ignore it
			continue
		}
		return false
	}

	return true
}
