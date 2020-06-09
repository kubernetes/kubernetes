package dbus

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
)

const defaultSystemBusAddress = "unix:path=/opt/local/var/run/dbus/system_bus_socket"

func getSessionBusPlatformAddress() (string, error) {
	cmd := exec.Command("launchctl", "getenv", "DBUS_LAUNCHD_SESSION_BUS_SOCKET")
	b, err := cmd.CombinedOutput()

	if err != nil {
		return "", err
	}

	if len(b) == 0 {
		return "", errors.New("dbus: couldn't determine address of session bus")
	}

	return "unix:path=" + string(b[:len(b)-1]), nil
}

func getSystemBusPlatformAddress() string {
	address := os.Getenv("DBUS_LAUNCHD_SESSION_BUS_SOCKET")
	if address != "" {
		return fmt.Sprintf("unix:path=%s", address)
	}
	return defaultSystemBusAddress
}

func tryDiscoverDbusSessionBusAddress() string {
	return ""
}
