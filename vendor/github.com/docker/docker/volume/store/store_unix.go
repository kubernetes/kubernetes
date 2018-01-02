// +build linux freebsd solaris

package store

// normaliseVolumeName is a platform specific function to normalise the name
// of a volume. This is a no-op on Unix-like platforms
func normaliseVolumeName(name string) string {
	return name
}
