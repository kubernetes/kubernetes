package daemon

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	containertypes "github.com/docker/docker/api/types/container"
	mounttypes "github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/container"
	"github.com/docker/docker/volume"
	"github.com/docker/docker/volume/drivers"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

var (
	// ErrVolumeReadonly is used to signal an error when trying to copy data into
	// a volume mount that is not writable.
	ErrVolumeReadonly = errors.New("mounted volume is marked read-only")
)

type mounts []container.Mount

// volumeToAPIType converts a volume.Volume to the type used by the Engine API
func volumeToAPIType(v volume.Volume) *types.Volume {
	createdAt, _ := v.CreatedAt()
	tv := &types.Volume{
		Name:      v.Name(),
		Driver:    v.DriverName(),
		CreatedAt: createdAt.Format(time.RFC3339),
	}
	if v, ok := v.(volume.DetailedVolume); ok {
		tv.Labels = v.Labels()
		tv.Options = v.Options()
		tv.Scope = v.Scope()
	}

	return tv
}

// Len returns the number of mounts. Used in sorting.
func (m mounts) Len() int {
	return len(m)
}

// Less returns true if the number of parts (a/b/c would be 3 parts) in the
// mount indexed by parameter 1 is less than that of the mount indexed by
// parameter 2. Used in sorting.
func (m mounts) Less(i, j int) bool {
	return m.parts(i) < m.parts(j)
}

// Swap swaps two items in an array of mounts. Used in sorting
func (m mounts) Swap(i, j int) {
	m[i], m[j] = m[j], m[i]
}

// parts returns the number of parts in the destination of a mount. Used in sorting.
func (m mounts) parts(i int) int {
	return strings.Count(filepath.Clean(m[i].Destination), string(os.PathSeparator))
}

// registerMountPoints initializes the container mount points with the configured volumes and bind mounts.
// It follows the next sequence to decide what to mount in each final destination:
//
// 1. Select the previously configured mount points for the containers, if any.
// 2. Select the volumes mounted from another containers. Overrides previously configured mount point destination.
// 3. Select the bind mounts set by the client. Overrides previously configured mount point destinations.
// 4. Cleanup old volumes that are about to be reassigned.
func (daemon *Daemon) registerMountPoints(container *container.Container, hostConfig *containertypes.HostConfig) (retErr error) {
	binds := map[string]bool{}
	mountPoints := map[string]*volume.MountPoint{}
	parser := volume.NewParser(container.OS)
	defer func() {
		// clean up the container mountpoints once return with error
		if retErr != nil {
			for _, m := range mountPoints {
				if m.Volume == nil {
					continue
				}
				daemon.volumes.Dereference(m.Volume, container.ID)
			}
		}
	}()

	dereferenceIfExists := func(destination string) {
		if v, ok := mountPoints[destination]; ok {
			logrus.Debugf("Duplicate mount point '%s'", destination)
			if v.Volume != nil {
				daemon.volumes.Dereference(v.Volume, container.ID)
			}
		}
	}

	// 1. Read already configured mount points.
	for destination, point := range container.MountPoints {
		mountPoints[destination] = point
	}

	// 2. Read volumes from other containers.
	for _, v := range hostConfig.VolumesFrom {
		containerID, mode, err := parser.ParseVolumesFrom(v)
		if err != nil {
			return err
		}

		c, err := daemon.GetContainer(containerID)
		if err != nil {
			return err
		}

		for _, m := range c.MountPoints {
			cp := &volume.MountPoint{
				Type:        m.Type,
				Name:        m.Name,
				Source:      m.Source,
				RW:          m.RW && parser.ReadWrite(mode),
				Driver:      m.Driver,
				Destination: m.Destination,
				Propagation: m.Propagation,
				Spec:        m.Spec,
				CopyData:    false,
			}

			if len(cp.Source) == 0 {
				v, err := daemon.volumes.GetWithRef(cp.Name, cp.Driver, container.ID)
				if err != nil {
					return err
				}
				cp.Volume = v
			}
			dereferenceIfExists(cp.Destination)
			mountPoints[cp.Destination] = cp
		}
	}

	// 3. Read bind mounts
	for _, b := range hostConfig.Binds {
		bind, err := parser.ParseMountRaw(b, hostConfig.VolumeDriver)
		if err != nil {
			return err
		}

		// #10618
		_, tmpfsExists := hostConfig.Tmpfs[bind.Destination]
		if binds[bind.Destination] || tmpfsExists {
			return duplicateMountPointError(bind.Destination)
		}

		if bind.Type == mounttypes.TypeVolume {
			// create the volume
			v, err := daemon.volumes.CreateWithRef(bind.Name, bind.Driver, container.ID, nil, nil)
			if err != nil {
				return err
			}
			bind.Volume = v
			bind.Source = v.Path()
			// bind.Name is an already existing volume, we need to use that here
			bind.Driver = v.DriverName()
			if bind.Driver == volume.DefaultDriverName {
				setBindModeIfNull(bind)
			}
		}

		binds[bind.Destination] = true
		dereferenceIfExists(bind.Destination)
		mountPoints[bind.Destination] = bind
	}

	for _, cfg := range hostConfig.Mounts {
		mp, err := parser.ParseMountSpec(cfg)
		if err != nil {
			return validationError{err}
		}

		if binds[mp.Destination] {
			return duplicateMountPointError(cfg.Target)
		}

		if mp.Type == mounttypes.TypeVolume {
			var v volume.Volume
			if cfg.VolumeOptions != nil {
				var driverOpts map[string]string
				if cfg.VolumeOptions.DriverConfig != nil {
					driverOpts = cfg.VolumeOptions.DriverConfig.Options
				}
				v, err = daemon.volumes.CreateWithRef(mp.Name, mp.Driver, container.ID, driverOpts, cfg.VolumeOptions.Labels)
			} else {
				v, err = daemon.volumes.CreateWithRef(mp.Name, mp.Driver, container.ID, nil, nil)
			}
			if err != nil {
				return err
			}

			mp.Volume = v
			mp.Name = v.Name()
			mp.Driver = v.DriverName()

			// only use the cached path here since getting the path is not necessary right now and calling `Path()` may be slow
			if cv, ok := v.(interface {
				CachedPath() string
			}); ok {
				mp.Source = cv.CachedPath()
			}
			if mp.Driver == volume.DefaultDriverName {
				setBindModeIfNull(mp)
			}
		}

		binds[mp.Destination] = true
		dereferenceIfExists(mp.Destination)
		mountPoints[mp.Destination] = mp
	}

	container.Lock()

	// 4. Cleanup old volumes that are about to be reassigned.
	for _, m := range mountPoints {
		if parser.IsBackwardCompatible(m) {
			if mp, exists := container.MountPoints[m.Destination]; exists && mp.Volume != nil {
				daemon.volumes.Dereference(mp.Volume, container.ID)
			}
		}
	}
	container.MountPoints = mountPoints

	container.Unlock()

	return nil
}

// lazyInitializeVolume initializes a mountpoint's volume if needed.
// This happens after a daemon restart.
func (daemon *Daemon) lazyInitializeVolume(containerID string, m *volume.MountPoint) error {
	if len(m.Driver) > 0 && m.Volume == nil {
		v, err := daemon.volumes.GetWithRef(m.Name, m.Driver, containerID)
		if err != nil {
			return err
		}
		m.Volume = v
	}
	return nil
}

// backportMountSpec resolves mount specs (introduced in 1.13) from pre-1.13
// mount configurations
// The container lock should not be held when calling this function.
// Changes are only made in-memory and may make changes to containers referenced
// by `container.HostConfig.VolumesFrom`
func (daemon *Daemon) backportMountSpec(container *container.Container) {
	container.Lock()
	defer container.Unlock()

	parser := volume.NewParser(container.OS)

	maybeUpdate := make(map[string]bool)
	for _, mp := range container.MountPoints {
		if mp.Spec.Source != "" && mp.Type != "" {
			continue
		}
		maybeUpdate[mp.Destination] = true
	}
	if len(maybeUpdate) == 0 {
		return
	}

	mountSpecs := make(map[string]bool, len(container.HostConfig.Mounts))
	for _, m := range container.HostConfig.Mounts {
		mountSpecs[m.Target] = true
	}

	binds := make(map[string]*volume.MountPoint, len(container.HostConfig.Binds))
	for _, rawSpec := range container.HostConfig.Binds {
		mp, err := parser.ParseMountRaw(rawSpec, container.HostConfig.VolumeDriver)
		if err != nil {
			logrus.WithError(err).Error("Got unexpected error while re-parsing raw volume spec during spec backport")
			continue
		}
		binds[mp.Destination] = mp
	}

	volumesFrom := make(map[string]volume.MountPoint)
	for _, fromSpec := range container.HostConfig.VolumesFrom {
		from, _, err := parser.ParseVolumesFrom(fromSpec)
		if err != nil {
			logrus.WithError(err).WithField("id", container.ID).Error("Error reading volumes-from spec during mount spec backport")
			continue
		}
		fromC, err := daemon.GetContainer(from)
		if err != nil {
			logrus.WithError(err).WithField("from-container", from).Error("Error looking up volumes-from container")
			continue
		}

		// make sure from container's specs have been backported
		daemon.backportMountSpec(fromC)

		fromC.Lock()
		for t, mp := range fromC.MountPoints {
			volumesFrom[t] = *mp
		}
		fromC.Unlock()
	}

	needsUpdate := func(containerMount, other *volume.MountPoint) bool {
		if containerMount.Type != other.Type || !reflect.DeepEqual(containerMount.Spec, other.Spec) {
			return true
		}
		return false
	}

	// main
	for _, cm := range container.MountPoints {
		if !maybeUpdate[cm.Destination] {
			continue
		}
		// nothing to backport if from hostconfig.Mounts
		if mountSpecs[cm.Destination] {
			continue
		}

		if mp, exists := binds[cm.Destination]; exists {
			if needsUpdate(cm, mp) {
				cm.Spec = mp.Spec
				cm.Type = mp.Type
			}
			continue
		}

		if cm.Name != "" {
			if mp, exists := volumesFrom[cm.Destination]; exists {
				if needsUpdate(cm, &mp) {
					cm.Spec = mp.Spec
					cm.Type = mp.Type
				}
				continue
			}

			if cm.Type != "" {
				// probably specified via the hostconfig.Mounts
				continue
			}

			// anon volume
			cm.Type = mounttypes.TypeVolume
			cm.Spec.Type = mounttypes.TypeVolume
		} else {
			if cm.Type != "" {
				// already updated
				continue
			}

			cm.Type = mounttypes.TypeBind
			cm.Spec.Type = mounttypes.TypeBind
			cm.Spec.Source = cm.Source
			if cm.Propagation != "" {
				cm.Spec.BindOptions = &mounttypes.BindOptions{
					Propagation: cm.Propagation,
				}
			}
		}

		cm.Spec.Target = cm.Destination
		cm.Spec.ReadOnly = !cm.RW
	}
}

func (daemon *Daemon) traverseLocalVolumes(fn func(volume.Volume) error) error {
	localVolumeDriver, err := volumedrivers.GetDriver(volume.DefaultDriverName)
	if err != nil {
		return fmt.Errorf("can't retrieve local volume driver: %v", err)
	}
	vols, err := localVolumeDriver.List()
	if err != nil {
		return fmt.Errorf("can't retrieve local volumes: %v", err)
	}

	for _, v := range vols {
		name := v.Name()
		vol, err := daemon.volumes.Get(name)
		if err != nil {
			logrus.Warnf("failed to retrieve volume %s from store: %v", name, err)
		} else {
			// daemon.volumes.Get will return DetailedVolume
			v = vol
		}

		err = fn(v)
		if err != nil {
			return err
		}
	}

	return nil
}
