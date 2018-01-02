// +build windows

package daemon

import (
	"sort"

	"github.com/docker/docker/container"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/volume"
)

// setupMounts configures the mount points for a container by appending each
// of the configured mounts on the container to the OCI mount structure
// which will ultimately be passed into the oci runtime during container creation.
// It also ensures each of the mounts are lexicographically sorted.

// BUGBUG TODO Windows containerd. This would be much better if it returned
// an array of runtime spec mounts, not container mounts. Then no need to
// do multiple transitions.

func (daemon *Daemon) setupMounts(c *container.Container) ([]container.Mount, error) {
	var mnts []container.Mount
	for _, mount := range c.MountPoints { // type is volume.MountPoint
		if err := daemon.lazyInitializeVolume(c.ID, mount); err != nil {
			return nil, err
		}
		s, err := mount.Setup(c.MountLabel, idtools.IDPair{0, 0}, nil)
		if err != nil {
			return nil, err
		}

		mnts = append(mnts, container.Mount{
			Source:      s,
			Destination: mount.Destination,
			Writable:    mount.RW,
		})
	}

	sort.Sort(mounts(mnts))
	return mnts, nil
}

// setBindModeIfNull is platform specific processing which is a no-op on
// Windows.
func setBindModeIfNull(bind *volume.MountPoint) {
	return
}
