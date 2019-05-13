package sdkuri

import (
	"path"
	"strings"
)

// PathJoin will join the elements of the path delimited by the "/"
// character. Similar to path.Join with the exception the trailing "/"
// character is preserved if present.
func PathJoin(elems ...string) string {
	if len(elems) == 0 {
		return ""
	}

	hasTrailing := strings.HasSuffix(elems[len(elems)-1], "/")
	str := path.Join(elems...)
	if hasTrailing && str != "/" {
		str += "/"
	}

	return str
}
