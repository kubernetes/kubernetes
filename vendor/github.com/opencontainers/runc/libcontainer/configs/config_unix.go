// +build freebsd linux

package configs

import "fmt"

// HostUID gets the root uid for the process on host which could be non-zero
// when user namespaces are enabled.
func (c Config) HostUID() (int, error) {
	if c.Namespaces.Contains(NEWUSER) {
		if c.UidMappings == nil {
			return -1, fmt.Errorf("User namespaces enabled, but no user mappings found.")
		}
		id, found := c.hostIDFromMapping(0, c.UidMappings)
		if !found {
			return -1, fmt.Errorf("User namespaces enabled, but no root user mapping found.")
		}
		return id, nil
	}
	// Return default root uid 0
	return 0, nil
}

// HostGID gets the root gid for the process on host which could be non-zero
// when user namespaces are enabled.
func (c Config) HostGID() (int, error) {
	if c.Namespaces.Contains(NEWUSER) {
		if c.GidMappings == nil {
			return -1, fmt.Errorf("User namespaces enabled, but no gid mappings found.")
		}
		id, found := c.hostIDFromMapping(0, c.GidMappings)
		if !found {
			return -1, fmt.Errorf("User namespaces enabled, but no root group mapping found.")
		}
		return id, nil
	}
	// Return default root gid 0
	return 0, nil
}

// Utility function that gets a host ID for a container ID from user namespace map
// if that ID is present in the map.
func (c Config) hostIDFromMapping(containerID int, uMap []IDMap) (int, bool) {
	for _, m := range uMap {
		if (containerID >= m.ContainerID) && (containerID <= (m.ContainerID + m.Size - 1)) {
			hostID := m.HostID + (containerID - m.ContainerID)
			return hostID, true
		}
	}
	return -1, false
}
