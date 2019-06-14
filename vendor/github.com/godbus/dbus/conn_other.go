// +build !darwin

package dbus

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
)

const defaultSystemBusAddress = "unix:path=/var/run/dbus/system_bus_socket"

func getSessionBusPlatformAddress() (string, error) {
	cmd := exec.Command("dbus-launch")
	b, err := cmd.CombinedOutput()

	if err != nil {
		return "", err
	}

	i := bytes.IndexByte(b, '=')
	j := bytes.IndexByte(b, '\n')

	if i == -1 || j == -1 {
		return "", errors.New("dbus: couldn't determine address of session bus")
	}

	env, addr := string(b[0:i]), string(b[i+1:j])
	os.Setenv(env, addr)

	return addr, nil
}

func getSystemBusPlatformAddress() string {
	address := os.Getenv("DBUS_SYSTEM_BUS_ADDRESS")
	if address != "" {
		return fmt.Sprintf("unix:path=%s", address)
	}
	return defaultSystemBusAddress
}
