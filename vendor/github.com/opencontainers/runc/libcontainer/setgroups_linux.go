// +build linux,go1.5

package libcontainer

import "syscall"

// Set the GidMappingsEnableSetgroups member to true, so the process's
// setgroups proc entry wont be set to 'deny' if GidMappings are set
func enableSetgroups(sys *syscall.SysProcAttr) {
	sys.GidMappingsEnableSetgroups = true
}
