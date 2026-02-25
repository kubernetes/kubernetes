//go:build !linux

package apparmor

func IsEnabled() bool {
	return false
}
