// +build !cgo,!plan9 windows android

package sftp

import (
	"os"
	"path"
)

func runLs(dirname string, dirent os.FileInfo) string {
	return path.Join(dirname, dirent.Name())
}
