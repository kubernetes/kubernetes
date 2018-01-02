package meta

import "os"

// renameFile will rename the source to target using os function. If target exists it will be removed before renaming.
func renameFile(oldpath, newpath string) error {
	if _, err := os.Stat(newpath); err == nil {
		if err = os.Remove(newpath); nil != err {
			return err
		}
	}

	return os.Rename(oldpath, newpath)
}
