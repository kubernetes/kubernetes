// +build !linux,!windows

package system

func ReadMemInfo() (*MemInfo, error) {
	return nil, ErrNotSupportedPlatform
}
