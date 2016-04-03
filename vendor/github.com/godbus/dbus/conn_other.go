// +build !darwin

package dbus

import (
	"bytes"
	"errors"
	"os/exec"
)

func sessionBusPlatform() (*Conn, error) {
	cmd := exec.Command("dbus-launch")
	b, err := cmd.CombinedOutput()

	if err != nil {
		return nil, err
	}

	i := bytes.IndexByte(b, '=')
	j := bytes.IndexByte(b, '\n')

	if i == -1 || j == -1 {
		return nil, errors.New("dbus: couldn't determine address of session bus")
	}

	return Dial(string(b[i+1 : j]))
}
