package fileutils

import (
	"os"
	"path/filepath"
)

// MkdirAllNewAs creates a directory (include any along the path) and then modifies
// ownership ONLY of newly created directories to the requested uid/gid. If the
// directories along the path exist, no change of ownership will be performed
func MkdirAllNewAs(path string, mode os.FileMode, ownerUID, ownerGID int) error {
	// make an array containing the original path asked for, plus (for mkAll == true)
	// all path components leading up to the complete path that don't exist before we MkdirAll
	// so that we can chown all of them properly at the end.  If chownExisting is false, we won't
	// chown the full directory path if it exists
	var paths []string
	if _, err := os.Stat(path); err != nil && os.IsNotExist(err) {
		paths = []string{path}
	} else if err == nil {
		// nothing to do; directory path fully exists already
		return nil
	}

	// walk back to "/" looking for directories which do not exist
	// and add them to the paths array for chown after creation
	dirPath := path
	for {
		dirPath = filepath.Dir(dirPath)
		if dirPath == "/" {
			break
		}
		if _, err := os.Stat(dirPath); err != nil && os.IsNotExist(err) {
			paths = append(paths, dirPath)
		}
	}

	if err := os.MkdirAll(path, mode); err != nil && !os.IsExist(err) {
		return err
	}

	// even if it existed, we will chown the requested path + any subpaths that
	// didn't exist when we called MkdirAll
	for _, pathComponent := range paths {
		if err := os.Chown(pathComponent, ownerUID, ownerGID); err != nil {
			return err
		}
	}
	return nil
}
