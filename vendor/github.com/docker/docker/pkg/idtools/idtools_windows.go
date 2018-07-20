package idtools

import (
	"os"

	"github.com/docker/docker/pkg/system"
)

// Platforms such as Windows do not support the UID/GID concept. So make this
// just a wrapper around system.MkdirAll.
func mkdirAs(path string, mode os.FileMode, ownerUID, ownerGID int, mkAll, chownExisting bool) error {
	if err := system.MkdirAll(path, mode, ""); err != nil {
		return err
	}
	return nil
}

// CanAccess takes a valid (existing) directory and a uid, gid pair and determines
// if that uid, gid pair has access (execute bit) to the directory
// Windows does not require/support this function, so always return true
func CanAccess(path string, pair IDPair) bool {
	return true
}
