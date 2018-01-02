// +build !windows

package dockerfile

import (
	"os"
	"path/filepath"

	"github.com/docker/docker/pkg/idtools"
)

func fixPermissions(source, destination string, rootIDs idtools.IDPair) error {
	skipChownRoot, err := isExistingDirectory(destination)
	if err != nil {
		return err
	}

	// We Walk on the source rather than on the destination because we don't
	// want to change permissions on things we haven't created or modified.
	return filepath.Walk(source, func(fullpath string, info os.FileInfo, err error) error {
		// Do not alter the walk root iff. it existed before, as it doesn't fall under
		// the domain of "things we should chown".
		if skipChownRoot && source == fullpath {
			return nil
		}

		// Path is prefixed by source: substitute with destination instead.
		cleaned, err := filepath.Rel(source, fullpath)
		if err != nil {
			return err
		}

		fullpath = filepath.Join(destination, cleaned)
		return os.Lchown(fullpath, rootIDs.UID, rootIDs.GID)
	})
}
