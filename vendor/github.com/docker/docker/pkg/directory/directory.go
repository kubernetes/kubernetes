package directory

import (
	"io/ioutil"
	"os"
	"path/filepath"
)

// MoveToSubdir moves all contents of a directory to a subdirectory underneath the original path
func MoveToSubdir(oldpath, subdir string) error {

	infos, err := ioutil.ReadDir(oldpath)
	if err != nil {
		return err
	}
	for _, info := range infos {
		if info.Name() != subdir {
			oldName := filepath.Join(oldpath, info.Name())
			newName := filepath.Join(oldpath, subdir, info.Name())
			if err := os.Rename(oldName, newName); err != nil {
				return err
			}
		}
	}
	return nil
}
