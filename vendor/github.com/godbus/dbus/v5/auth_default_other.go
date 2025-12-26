//go:build !windows
// +build !windows

package dbus

func getDefaultAuthMethods(user string) []Auth {
	return []Auth{AuthExternal(user)}
}
