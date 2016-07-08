// +build windows

package utils

// HostName returns then host name.
func HostName() (string, error) {
	return "windows", nil
}
