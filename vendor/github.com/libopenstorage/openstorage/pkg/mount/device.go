// +build linux

package mount

import (
	"strings"

	"github.com/docker/docker/pkg/mount"
)

// DeviceMounter implements Ops and tracks active mounts for volume drivers.
type DeviceMounter struct {
	Mounter
}

// NewDeviceMounter returns a new DeviceMounter
func NewDeviceMounter(
	devPrefixes []string,
	mountImpl MountImpl,
	allowedDirs []string,
) (*DeviceMounter, error) {

	m := &DeviceMounter{
		Mounter: Mounter{
			mountImpl:   mountImpl,
			mounts:      make(DeviceMap),
			paths:       make(PathMap),
			allowedDirs: allowedDirs,
		},
	}
	err := m.Load(devPrefixes)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// Reload reloads the mount table
func (m *DeviceMounter) Reload(device string) error {
	newDm, err := NewDeviceMounter([]string{device}, m.mountImpl, m.Mounter.allowedDirs)
	if err != nil {
		return err
	}
	m.Lock()
	defer m.Unlock()

	// New mountable has no mounts, delete old mounts.
	newM, ok := newDm.mounts[device]
	if !ok {
		delete(m.mounts, device)
		return nil
	}

	// Old mountable had no mounts, copy over new mounts.
	oldM, ok := m.mounts[device]
	if !ok {
		m.mounts[device] = newM
		return nil
	}

	// Overwrite old mount entries into new mount table, preserving refcnt.
	for _, oldP := range oldM.Mountpoint {
		for j, newP := range newM.Mountpoint {
			if newP.Path == oldP.Path {
				newM.Mountpoint[j] = oldP
				break
			}
		}
	}

	// Purge old mounts.
	m.mounts[device] = newM
	return nil
}

// Load mount table
func (m *DeviceMounter) Load(devPrefixes []string) error {
	info, err := mount.GetMounts()
	if err != nil {
		return err
	}
DeviceLoop:
	for _, v := range info {
		foundPrefix := false
		for _, devPrefix := range devPrefixes {
			if strings.HasPrefix(v.Source, devPrefix) {
				foundPrefix = true
				break
			}
		}
		if !foundPrefix {
			continue
		}
		mount, ok := m.mounts[v.Source]
		if !ok {
			mount = &Info{
				Device:     v.Source,
				Fs:         v.Fstype,
				Minor:      v.Minor,
				Mountpoint: make([]*PathInfo, 0),
			}
			m.mounts[v.Source] = mount
		}
		// Allow Load to be called multiple times.
		for _, p := range mount.Mountpoint {
			if p.Path == v.Mountpoint {
				continue DeviceLoop
			}
		}
		mount.Mountpoint = append(
			mount.Mountpoint,
			&PathInfo{
				Path: normalizeMountPath(v.Mountpoint),
			},
		)
		m.paths[v.Mountpoint] = v.Source
	}
	return nil
}
