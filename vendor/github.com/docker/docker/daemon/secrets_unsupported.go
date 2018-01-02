// +build !linux,!windows

package daemon

func secretsSupported() bool {
	return false
}
