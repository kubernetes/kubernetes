package configs

import (
	"errors"
	"fmt"
	"math"
)

var (
	errNoUIDMap   = errors.New("User namespaces enabled, but no uid mappings found.")
	errNoUserMap  = errors.New("User namespaces enabled, but no user mapping found.")
	errNoGIDMap   = errors.New("User namespaces enabled, but no gid mappings found.")
	errNoGroupMap = errors.New("User namespaces enabled, but no group mapping found.")
)

// HostUID gets the translated uid for the process on host which could be
// different when user namespaces are enabled.
func (c Config) HostUID(containerId int) (int, error) {
	if c.Namespaces.Contains(NEWUSER) {
		if c.UidMappings == nil {
			return -1, errNoUIDMap
		}
		id, found := c.hostIDFromMapping(int64(containerId), c.UidMappings)
		if !found {
			return -1, errNoUserMap
		}
		// If we are a 32-bit binary running on a 64-bit system, it's possible
		// the mapped user is too large to store in an int, which means we
		// cannot do the mapping. We can't just return an int64, because
		// os.Setuid() takes an int.
		if id > math.MaxInt {
			return -1, fmt.Errorf("mapping for uid %d (host id %d) is larger than native integer size (%d)", containerId, id, math.MaxInt)
		}
		return int(id), nil
	}
	// Return unchanged id.
	return containerId, nil
}

// HostRootUID gets the root uid for the process on host which could be non-zero
// when user namespaces are enabled.
func (c Config) HostRootUID() (int, error) {
	return c.HostUID(0)
}

// HostGID gets the translated gid for the process on host which could be
// different when user namespaces are enabled.
func (c Config) HostGID(containerId int) (int, error) {
	if c.Namespaces.Contains(NEWUSER) {
		if c.GidMappings == nil {
			return -1, errNoGIDMap
		}
		id, found := c.hostIDFromMapping(int64(containerId), c.GidMappings)
		if !found {
			return -1, errNoGroupMap
		}
		// If we are a 32-bit binary running on a 64-bit system, it's possible
		// the mapped user is too large to store in an int, which means we
		// cannot do the mapping. We can't just return an int64, because
		// os.Setgid() takes an int.
		if id > math.MaxInt {
			return -1, fmt.Errorf("mapping for gid %d (host id %d) is larger than native integer size (%d)", containerId, id, math.MaxInt)
		}
		return int(id), nil
	}
	// Return unchanged id.
	return containerId, nil
}

// HostRootGID gets the root gid for the process on host which could be non-zero
// when user namespaces are enabled.
func (c Config) HostRootGID() (int, error) {
	return c.HostGID(0)
}

// Utility function that gets a host ID for a container ID from user namespace map
// if that ID is present in the map.
func (c Config) hostIDFromMapping(containerID int64, uMap []IDMap) (int64, bool) {
	for _, m := range uMap {
		if (containerID >= m.ContainerID) && (containerID <= (m.ContainerID + m.Size - 1)) {
			hostID := m.HostID + (containerID - m.ContainerID)
			return hostID, true
		}
	}
	return -1, false
}
