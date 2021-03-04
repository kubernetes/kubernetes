// +build !static_build

package dbus

import (
	"os/user"
)

func lookupHomeDir() string {
	u, err := user.Current()
	if err != nil {
		return "/"
	}
	return u.HomeDir
}
