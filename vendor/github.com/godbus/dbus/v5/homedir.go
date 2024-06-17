package dbus

import (
	"os"
	"os/user"
)

// Get returns the home directory of the current user, which is usually the
// value of HOME environment variable. In case it is not set or empty, os/user
// package is used.
//
// If linking statically with cgo enabled against glibc, make sure the
// osusergo build tag is used.
//
// If needing to do nss lookups, do not disable cgo or set osusergo.
func getHomeDir() string {
	homeDir := os.Getenv("HOME")
	if homeDir != "" {
		return homeDir
	}
	if u, err := user.Current(); err == nil {
		return u.HomeDir
	}
	return "/"
}
