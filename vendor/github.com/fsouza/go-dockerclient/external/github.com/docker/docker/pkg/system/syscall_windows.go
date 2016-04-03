package system

import (
	"fmt"
	"syscall"
)

// OSVersion is a wrapper for Windows version information
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms724439(v=vs.85).aspx
type OSVersion struct {
	Version      uint32
	MajorVersion uint8
	MinorVersion uint8
	Build        uint16
}

// GetOSVersion gets the operating system version on Windows. Note that
// docker.exe must be manifested to get the correct version information.
func GetOSVersion() (OSVersion, error) {
	var err error
	osv := OSVersion{}
	osv.Version, err = syscall.GetVersion()
	if err != nil {
		return osv, fmt.Errorf("Failed to call GetVersion()")
	}
	osv.MajorVersion = uint8(osv.Version & 0xFF)
	osv.MinorVersion = uint8(osv.Version >> 8 & 0xFF)
	osv.Build = uint16(osv.Version >> 16)
	return osv, nil
}

// Unmount is a platform-specific helper function to call
// the unmount syscall. Not supported on Windows
func Unmount(dest string) error {
	return nil
}
