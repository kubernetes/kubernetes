// +build !linux

package system

func ReadMemInfo() (*MemInfo, error) {
	return nil, ErrNotSupportedPlatform
}
