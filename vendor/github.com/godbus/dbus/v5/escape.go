package dbus

import "net/url"

// EscapeBusAddressValue implements a requirement to escape the values
// in D-Bus server addresses, as defined by the D-Bus specification at
// https://dbus.freedesktop.org/doc/dbus-specification.html#addresses.
func EscapeBusAddressValue(val string) string {
	toEsc := strNeedsEscape(val)
	if toEsc == 0 {
		// Avoid unneeded allocation/copying.
		return val
	}

	// Avoid allocation for short paths.
	var buf [64]byte
	var out []byte
	// Every to-be-escaped byte needs 2 extra bytes.
	required := len(val) + 2*toEsc
	if required <= len(buf) {
		out = buf[:required]
	} else {
		out = make([]byte, required)
	}

	j := 0
	for i := 0; i < len(val); i++ {
		if ch := val[i]; needsEscape(ch) {
			// Convert ch to %xx, where xx is hex value.
			out[j] = '%'
			out[j+1] = hexchar(ch >> 4)
			out[j+2] = hexchar(ch & 0x0F)
			j += 3
		} else {
			out[j] = ch
			j++
		}
	}

	return string(out)
}

// UnescapeBusAddressValue unescapes values in D-Bus server addresses,
// as defined by the D-Bus specification at
// https://dbus.freedesktop.org/doc/dbus-specification.html#addresses.
func UnescapeBusAddressValue(val string) (string, error) {
	// Looks like url.PathUnescape does exactly what is required.
	return url.PathUnescape(val)
}

// hexchar returns an octal representation of a n, where n < 16.
// For invalid values of n, the function panics.
func hexchar(n byte) byte {
	const hex = "0123456789abcdef"

	// For n >= len(hex), runtime will panic.
	return hex[n]
}

// needsEscape tells if a byte is NOT one of optionally-escaped bytes.
func needsEscape(c byte) bool {
	if 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9' {
		return false
	}
	switch c {
	case '-', '_', '/', '\\', '.', '*':
		return false
	}

	return true
}

// strNeedsEscape tells how many bytes in the string need escaping.
func strNeedsEscape(val string) int {
	count := 0

	for i := 0; i < len(val); i++ {
		if needsEscape(val[i]) {
			count++
		}
	}

	return count
}
