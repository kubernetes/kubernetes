// +build appengine

package msgp

// let's just assume appengine
// uses 64-bit hardware...
const smallint = false

func UnsafeString(b []byte) string {
	return string(b)
}

func UnsafeBytes(s string) []byte {
	return []byte(s)
}
