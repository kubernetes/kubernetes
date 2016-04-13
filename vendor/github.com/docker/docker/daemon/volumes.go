package daemon

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/runconfig"
	"github.com/docker/docker/volume"
	"github.com/docker/docker/volume/drivers"
	"github.com/docker/docker/volume/local"
	"github.com/opencontainers/runc/libcontainer/label"
)

// ErrVolumeReadonly is used to signal an error when trying to copy data into
// a volume mount that is not writable.
var ErrVolumeReadonly = errors.New("mounted volume is marked read-only")

type mountPoint struct {
	Name        string
	Destination string
	Driver      string
	RW          bool
	Volume      volume.Volume `json:"-"`
	Source      string
	Relabel     string
}

func (m *mountPoint) Setup() (string, error) {
	if m.Volume != nil {
		return m.Volume.Mount()
	}

	if len(m.Source) > 0 {
		if _, err := os.Stat(m.Source); err != nil {
			if !os.IsNotExist(err) {
				return "", err
			}
			if err := system.MkdirAll(m.Source, 0755); err != nil {
				return "", err
			}
		}
		return m.Source, nil
	}

	return "", fmt.Errorf("Unable to setup mount point, neither source nor volume defined")
}

// hasResource checks whether the given absolute path for a container is in
// this mount point. If the relative path starts with `../` then the resource
// is outside of this mount point, but we can't simply check for this prefix
// because it misses `..` which is also outside of the mount, so check both.
func (m *mountPoint) hasResource(absolutePath string) bool {
	relPath, err := filepath.Rel(m.Destination, absolutePath)

	return err == nil && relPath != ".." && !strings.HasPrefix(relPath, fmt.Sprintf("..%c", filepath.Separator))
}

func (m *mountPoint) Path() string {
	if m.Volume != nil {
		return m.Volume.Path()
	}

	return m.Source
}

// BackwardsCompatible decides whether this mount point can be
// used in old versions of Docker or not.
// Only bind mounts and local volumes can be used in old versions of Docker.
func (m *mountPoint) BackwardsCompatible() bool {
	return len(m.Source) > 0 || m.Driver == volume.DefaultDriverName
}

func parseBindMount(spec string, mountLabel string, config *runconfig.Config) (*mountPoint, error) {
	bind := &mountPoint{
		RW: true,
	}
	arr := strings.Split(spec, ":")

	switch len(arr) {
	case 2:
		bind.Destination = arr[1]
	case 3:
		bind.Destination = arr[1]
		mode := arr[2]
		isValid, isRw := volume.ValidateMountMode(mode)
		if !isValid {
			return nil, fmt.Errorf("invalid mode for volumes-from: %s", mode)
		}
		bind.RW = isRw
		// Relabel will apply a SELinux label, if necessary
		bind.Relabel = mode
	default:
		return nil, fmt.Errorf("Invalid volume specification: %s", spec)
	}

	name, source, err := parseVolumeSource(arr[0])
	if err != nil {
		return nil, err
	}

	if len(source) == 0 {
		bind.Driver = config.VolumeDriver
		if len(bind.Driver) == 0 {
			bind.Driver = volume.DefaultDriverName
		}
	} else {
		bind.Source = filepath.Clean(source)
	}

	bind.Name = name
	bind.Destination = filepath.Clean(bind.Destination)
	return bind, nil
}

func parseVolumesFrom(spec string) (string, string, error) {
	if len(spec) == 0 {
		return "", "", fmt.Errorf("malformed volumes-from specification: %s", spec)
	}

	specParts := strings.SplitN(spec, ":", 2)
	id := specParts[0]
	mode := "rw"

	if len(specParts) == 2 {
		mode = specParts[1]
		if isValid, _ := volume.ValidateMountMode(mode); !isValid {
			return "", "", fmt.Errorf("invalid mode for volumes-from: %s", mode)
		}
	}
	return id, mode, nil
}

func copyExistingContents(source, destination string) error {
	volList, err := ioutil.ReadDir(source)
	if err != nil {
		return err
	}
	if len(volList) > 0 {
		srcList, err := ioutil.ReadDir(destination)
		if err != nil {
			return err
		}
		if len(srcList) == 0 {
			// If the source volume is empty copy files from the root into the volume
			if err := chrootarchive.CopyWithTar(source, destination); err != nil {
				return err
			}
		}
	}
	return copyOwnership(source, destination)
}

// registerMountPoints initializes the container mount points with the configured volumes and bind mounts.
// It follows the next sequence to decide what to mount in each final destination:
//
// 1. Select the previously configured mount points for the containers, if any.
// 2. Select the volumes mounted from another containers. Overrides previously configured mount point destination.
// 3. Select the bind mounts set by the client. Overrides previously configured mount point destinations.
func (daemon *Daemon) registerMountPoints(container *Container, hostConfig *runconfig.HostConfig) error {
	binds := map[string]bool{}
	mountPoints := map[string]*mountPoint{}

	// 1. Read already configured mount points.
	for name, point := range container.MountPoints {
		mountPoints[name] = point
	}

	// 2. Read volumes from other containers.
	for _, v := range hostConfig.VolumesFrom {
		containerID, mode, err := parseVolumesFrom(v)
		if err != nil {
			return err
		}

		c, err := daemon.Get(containerID)
		if err != nil {
			return err
		}

		for _, m := range c.MountPoints {
			cp := &mountPoint{
				Name:        m.Name,
				Source:      m.Source,
				RW:          m.RW && volume.ReadWrite(mode),
				Driver:      m.Driver,
				Destination: m.Destination,
			}

			if len(cp.Source) == 0 {
				v, err := createVolume(cp.Name, cp.Driver)
				if err != nil {
					return err
				}
				cp.Volume = v
			}

			mountPoints[cp.Destination] = cp
		}
	}

	// 3. Read bind mounts
	for _, b := range hostConfig.Binds {
		// #10618
		bind, err := parseBindMount(b, container.MountLabel, container.Config)
		if err != nil {
			return err
		}

		if binds[bind.Destination] {
			return fmt.Errorf("Duplicate bind mount %s", bind.Destination)
		}

		if len(bind.Name) > 0 && len(bind.Driver) > 0 {
			// create the volume
			v, err := createVolume(bind.Name, bind.Driver)
			if err != nil {
				return err
			}
			bind.Volume = v
			bind.Source = v.Path()
			// Since this is just a named volume and not a typical bind, set to shared mode `z`
			if bind.Relabel == "" {
				bind.Relabel = "z"
			}
		}

		if err := label.Relabel(bind.Source, container.MountLabel, bind.Relabel); err != nil {
			return err
		}
		binds[bind.Destination] = true
		mountPoints[bind.Destination] = bind
	}

	// Keep backwards compatible structures
	bcVolumes := map[string]string{}
	bcVolumesRW := map[string]bool{}
	for _, m := range mountPoints {
		if m.BackwardsCompatible() {
			bcVolumes[m.Destination] = m.Path()
			bcVolumesRW[m.Destination] = m.RW
		}
	}

	container.Lock()
	container.MountPoints = mountPoints
	container.Volumes = bcVolumes
	container.VolumesRW = bcVolumesRW
	container.Unlock()

	return nil
}

// TODO Windows. Factor out as not relevant (as Windows daemon support not in pre-1.7)
// verifyVolumesInfo ports volumes configured for the containers pre docker 1.7.
// It reads the container configuration and creates valid mount points for the old volumes.
func (daemon *Daemon) verifyVolumesInfo(container *Container) error {
	// Inspect old structures only when we're upgrading from old versions
	// to versions >= 1.7 and the MountPoints has not been populated with volumes data.
	if len(container.MountPoints) == 0 && len(container.Volumes) > 0 {
		for destination, hostPath := range container.Volumes {
			vfsPath := filepath.Join(daemon.root, "vfs", "dir")
			rw := container.VolumesRW != nil && container.VolumesRW[destination]

			if strings.HasPrefix(hostPath, vfsPath) {
				id := filepath.Base(hostPath)
				if err := migrateVolume(id, hostPath); err != nil {
					return err
				}
				container.addLocalMountPoint(id, destination, rw)
			} else { // Bind mount
				id, source, err := parseVolumeSource(hostPath)
				// We should not find an error here coming
				// from the old configuration, but who knows.
				if err != nil {
					return err
				}
				container.addBindMountPoint(id, source, destination, rw)
			}
		}
	} else if len(container.MountPoints) > 0 {
		// Volumes created with a Docker version >= 1.7. We verify integrity in case of data created
		// with Docker 1.7 RC versions that put the information in
		// DOCKER_ROOT/volumes/VOLUME_ID rather than DOCKER_ROOT/volumes/VOLUME_ID/_container_data.
		l, err := getVolumeDriver(volume.DefaultDriverName)
		if err != nil {
			return err
		}

		for _, m := range container.MountPoints {
			if m.Driver != volume.DefaultDriverName {
				continue
			}
			dataPath := l.(*local.Root).DataPath(m.Name)
			volumePath := filepath.Dir(dataPath)

			d, err := ioutil.ReadDir(volumePath)
			if err != nil {
				// If the volume directory doesn't exist yet it will be recreated,
				// so we only return the error when there is a different issue.
				if !os.IsNotExist(err) {
					return err
				}
				// Do not check when the volume directory does not exist.
				continue
			}
			if validVolumeLayout(d) {
				continue
			}

			if err := os.Mkdir(dataPath, 0755); err != nil {
				return err
			}

			// Move data inside the data directory
			for _, f := range d {
				oldp := filepath.Join(volumePath, f.Name())
				newp := filepath.Join(dataPath, f.Name())
				if err := os.Rename(oldp, newp); err != nil {
					logrus.Errorf("Unable to move %s to %s\n", oldp, newp)
				}
			}
		}

		return container.ToDisk()
	}

	return nil
}

func createVolume(name, driverName string) (volume.Volume, error) {
	vd, err := getVolumeDriver(driverName)
	if err != nil {
		return nil, err
	}
	return vd.Create(name)
}

func removeVolume(v volume.Volume) error {
	vd, err := getVolumeDriver(v.DriverName())
	if err != nil {
		return nil
	}
	return vd.Remove(v)
}

func getVolumeDriver(name string) (volume.Driver, error) {
	if name == "" {
		name = volume.DefaultDriverName
	}
	return volumedrivers.Lookup(name)
}

func parseVolumeSource(spec string) (string, string, error) {
	if !filepath.IsAbs(spec) {
		return spec, "", nil
	}

	return "", spec, nil
}
