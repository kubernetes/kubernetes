// +build !linux,!windows

package daemon

func configsSupported() bool {
	return false
}
