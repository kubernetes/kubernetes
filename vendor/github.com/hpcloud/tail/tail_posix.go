// +build linux darwin freebsd netbsd openbsd

package tail

import (
	"os"
)

func OpenFile(name string) (file *os.File, err error) {
	return os.Open(name)
}
