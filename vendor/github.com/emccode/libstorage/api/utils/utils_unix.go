// +build !windows

package utils

import "os/exec"

const (
	newline = 10
)

// HostName returns then host name.
func HostName() (string, error) {
	buf, err := exec.Command("hostname").Output()
	if err != nil {
		return "", err
	}
	if buf[len(buf)-1] == 10 {
		buf = buf[:len(buf)-1]
	}
	return string(buf), nil
}
