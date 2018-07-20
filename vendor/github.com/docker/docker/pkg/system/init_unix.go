// +build !windows

package system

// InitLCOW does nothing since LCOW is a windows only feature
func InitLCOW(experimental bool) {
}
