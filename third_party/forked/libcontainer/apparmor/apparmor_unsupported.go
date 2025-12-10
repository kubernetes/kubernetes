//go:build !linux
// +build !linux

package apparmor

func IsEnabled() bool {
	return false
}
