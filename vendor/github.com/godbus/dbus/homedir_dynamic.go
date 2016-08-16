// +build !static_build

package dbus

import (
	"runtime"
	"os/user"
)

func lookupHomeDir() string {
	runtime.LockOSThread()
	u, err := user.Current()
	runtime.UnlockOSThread()
	if err != nil {
		return guessHomeDir()
	}
	return u.HomeDir
}
