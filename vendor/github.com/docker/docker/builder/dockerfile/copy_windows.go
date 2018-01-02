package dockerfile

import "github.com/docker/docker/pkg/idtools"

func fixPermissions(source, destination string, rootIDs idtools.IDPair) error {
	// chown is not supported on Windows
	return nil
}
