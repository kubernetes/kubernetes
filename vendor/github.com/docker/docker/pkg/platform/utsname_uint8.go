// +build linux,arm linux,ppc64 linux,ppc64le
// see golang's sources golang.org/x/sys/unix/ztypes_linux_*.go that use uint8

package platform

// Convert the OS/ARCH-specific utsname.Machine to string
// given as an array of unsigned uint8
func charsToString(ca [65]uint8) string {
	s := make([]byte, len(ca))
	var lens int
	for ; lens < len(ca); lens++ {
		if ca[lens] == 0 {
			break
		}
		s[lens] = ca[lens]
	}
	return string(s[0:lens])
}
