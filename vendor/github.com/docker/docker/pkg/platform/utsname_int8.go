// +build linux,386 linux,amd64 linux,arm64 s390x
// see golang's sources golang.org/x/sys/unix/ztypes_linux_*.go that use int8

package platform

// Convert the OS/ARCH-specific utsname.Machine to string
// given as an array of signed int8
func charsToString(ca [65]int8) string {
	s := make([]byte, len(ca))
	var lens int
	for ; lens < len(ca); lens++ {
		if ca[lens] == 0 {
			break
		}
		s[lens] = uint8(ca[lens])
	}
	return string(s[0:lens])
}
