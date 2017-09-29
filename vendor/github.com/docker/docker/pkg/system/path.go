package system

import "runtime"

const defaultUnixPathEnv = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

// DefaultPathEnv is unix style list of directories to search for
// executables. Each directory is separated from the next by a colon
// ':' character .
func DefaultPathEnv(platform string) string {
	if runtime.GOOS == "windows" {
		if platform != runtime.GOOS && LCOWSupported() {
			return defaultUnixPathEnv
		}
		// Deliberately empty on Windows containers on Windows as the default path will be set by
		// the container. Docker has no context of what the default path should be.
		return ""
	}
	return defaultUnixPathEnv

}
