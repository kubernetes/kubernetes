package dbus

import (
	"errors"
	"os/exec"
)

func sessionBusPlatform() (*Conn, error) {
	cmd := exec.Command("launchctl", "getenv", "DBUS_LAUNCHD_SESSION_BUS_SOCKET")
	b, err := cmd.CombinedOutput()

	if err != nil {
		return nil, err
	}

	if len(b) == 0 {
		return nil, errors.New("dbus: couldn't determine address of session bus")
	}

	return Dial("unix:path=" + string(b[:len(b)-1]))
}
