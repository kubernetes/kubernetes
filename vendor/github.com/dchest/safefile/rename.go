// +build !plan9,!windows windows,go1.5

package safefile

import "os"

func rename(oldname, newname string) error {
	return os.Rename(oldname, newname)
}
