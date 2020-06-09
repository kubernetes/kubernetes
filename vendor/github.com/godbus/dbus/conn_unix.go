//+build !windows,!solaris,!darwin

package dbus

import (
	"os"
	"fmt"
)

const defaultSystemBusAddress = "unix:path=/var/run/dbus/system_bus_socket"

func getSystemBusPlatformAddress() string {
	address := os.Getenv("DBUS_SYSTEM_BUS_ADDRESS")
	if address != "" {
		return fmt.Sprintf("unix:path=%s", address)
	}
	return defaultSystemBusAddress
}