package nlenc

import "bytes"

// Bytes returns a null-terminated byte slice with the contents of s.
func Bytes(s string) []byte {
	return append([]byte(s), 0x00)
}

// String returns a string with the contents of b from a null-terminated
// byte slice.
func String(b []byte) string {
	// If the string has more than one NULL terminator byte, we want to remove
	// all of them before returning the string to the caller; hence the use of
	// strings.TrimRight instead of strings.TrimSuffix (which previously only
	// removed a single NULL).
	return string(bytes.TrimRight(b, "\x00"))
}
