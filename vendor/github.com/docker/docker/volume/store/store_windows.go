package store

import "strings"

// normaliseVolumeName is a platform specific function to normalise the name
// of a volume. On Windows, as NTFS is case insensitive, under
// c:\ProgramData\Docker\Volumes\, the folders John and john would be synonymous.
// Hence we can't allow the volume "John" and "john" to be created as separate
// volumes.
func normaliseVolumeName(name string) string {
	return strings.ToLower(name)
}
