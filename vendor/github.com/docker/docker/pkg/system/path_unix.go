// +build !windows

package system

// CheckSystemDriveAndRemoveDriveLetter verifies that a path, if it includes a drive letter,
// is the system drive. This is a no-op on Linux.
func CheckSystemDriveAndRemoveDriveLetter(path string) (string, error) {
	return path, nil
}
