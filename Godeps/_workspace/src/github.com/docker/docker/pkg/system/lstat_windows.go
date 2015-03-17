// +build windows

package system

func Lstat(path string) (*Stat_t, error) {
	// should not be called on cli code path
	return nil, ErrNotSupportedPlatform
}
