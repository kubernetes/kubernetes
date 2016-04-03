// +build go1.4

package libcontainer

import "syscall"

// Converts IDMap to SysProcIDMap array and adds it to SysProcAttr.
func (c *linuxContainer) addUidGidMappings(sys *syscall.SysProcAttr) error {
	if c.config.UidMappings != nil {
		sys.UidMappings = make([]syscall.SysProcIDMap, len(c.config.UidMappings))
		for i, um := range c.config.UidMappings {
			sys.UidMappings[i].ContainerID = um.ContainerID
			sys.UidMappings[i].HostID = um.HostID
			sys.UidMappings[i].Size = um.Size
		}
	}
	if c.config.GidMappings != nil {
		sys.GidMappings = make([]syscall.SysProcIDMap, len(c.config.GidMappings))
		for i, gm := range c.config.GidMappings {
			sys.GidMappings[i].ContainerID = gm.ContainerID
			sys.GidMappings[i].HostID = gm.HostID
			sys.GidMappings[i].Size = gm.Size
		}
	}
	return nil
}
