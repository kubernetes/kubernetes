// longpath introduces some constants and helper functions for handling long paths
// in Windows, which are expected to be prepended with `\\?\` and followed by either
// a drive letter, a UNC server\share, or a volume identifier.

package longpath

import (
	"strings"
)

// Prefix is the longpath prefix for Windows file paths.
const Prefix = `\\?\`

// AddPrefix will add the Windows long path prefix to the path provided if
// it does not already have it.
func AddPrefix(path string) string {
	if !strings.HasPrefix(path, Prefix) {
		if strings.HasPrefix(path, `\\`) {
			// This is a UNC path, so we need to add 'UNC' to the path as well.
			path = Prefix + `UNC` + path[1:]
		} else {
			path = Prefix + path
		}
	}
	return path
}
