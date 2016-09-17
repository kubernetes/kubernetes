// +build windows

package idtools

import (
	"os"

	"github.com/docker/docker/pkg/system"
)

// Platforms such as Windows do not support the UID/GID concept. So make this
// just a wrapper around system.MkdirAll.
func mkdirAs(path string, mode os.FileMode, ownerUID, ownerGID int, mkAll, chownExisting bool) error {
	if err := system.MkdirAll(path, mode); err != nil && !os.IsExist(err) {
		return err
	}
	return nil
}
