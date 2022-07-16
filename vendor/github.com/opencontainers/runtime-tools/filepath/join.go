package filepath

import "strings"

// Join is an explicit-OS version of path/filepath's Join.
func Join(os string, elem ...string) string {
	sep := Separator(os)
	return Clean(os, strings.Join(elem, string(sep)))
}
