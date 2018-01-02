package daemon

import (
	"fmt"
	"strings"

	mounttypes "github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/container"
	volumestore "github.com/docker/docker/volume/store"
)

func (daemon *Daemon) prepareMountPoints(container *container.Container) error {
	for _, config := range container.MountPoints {
		if err := daemon.lazyInitializeVolume(container.ID, config); err != nil {
			return err
		}
	}
	return nil
}

func (daemon *Daemon) removeMountPoints(container *container.Container, rm bool) error {
	var rmErrors []string
	for _, m := range container.MountPoints {
		if m.Type != mounttypes.TypeVolume || m.Volume == nil {
			continue
		}
		daemon.volumes.Dereference(m.Volume, container.ID)
		if !rm {
			continue
		}

		// Do not remove named mountpoints
		// these are mountpoints specified like `docker run -v <name>:/foo`
		if m.Spec.Source != "" {
			continue
		}

		err := daemon.volumes.Remove(m.Volume)
		// Ignore volume in use errors because having this
		// volume being referenced by other container is
		// not an error, but an implementation detail.
		// This prevents docker from logging "ERROR: Volume in use"
		// where there is another container using the volume.
		if err != nil && !volumestore.IsInUse(err) {
			rmErrors = append(rmErrors, err.Error())
		}
	}

	if len(rmErrors) > 0 {
		return fmt.Errorf("Error removing volumes:\n%v", strings.Join(rmErrors, "\n"))
	}
	return nil
}
