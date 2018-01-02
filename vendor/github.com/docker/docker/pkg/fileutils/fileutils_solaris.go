package fileutils

// GetTotalUsedFds Returns the number of used File Descriptors.
// On Solaris these limits are per process and not systemwide
func GetTotalUsedFds() int {
	return -1
}
