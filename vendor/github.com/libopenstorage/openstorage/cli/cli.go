package cli

import (
	"github.com/codegangsta/cli"
)

const (
	// DaemonAlias command aliases for daemon mode.
	DaemonAlias = "daemon, d"
	// DaemonFlag key for the daeomon parameter
	DaemonFlag = "daemon"
	// DriverFlag key for for the driver parameter.
	DriverFlag = "driver"
)

// DaemonMode returns true if we are running as daemon
func DaemonMode(c *cli.Context) bool {
	return c.GlobalBool(DaemonFlag)
}

// DriverName as specified in the -<DriverFlag> parameter.
func DriverName(c *cli.Context) string {
	return c.GlobalString(DriverFlag)
}
