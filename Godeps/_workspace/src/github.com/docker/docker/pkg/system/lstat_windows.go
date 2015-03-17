// +build windows

package system

func Lstat(path string) (*Stat, error) {
	// should not be called on cli code path
	return nil, ErrNotSupportedPlatform
}
