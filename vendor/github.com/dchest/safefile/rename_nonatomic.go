// +build plan9 windows,!go1.5

// os.Rename on Windows before Go 1.5 and Plan 9 will not overwrite existing
// files, thus we cannot guarantee atomic saving of file by doing rename.
// We will have to do some voodoo to minimize data loss on those systems.

package safefile

import (
	"os"
	"path/filepath"
)

func rename(oldname, newname string) error {
	err := os.Rename(oldname, newname)
	if err != nil {
		// If newname exists ("original"), we will try renaming it to a
		// new temporary name, then renaming oldname to the newname,
		// and deleting the renamed original. If system crashes between
		// renaming and deleting, the original file will still be available
		// under the temporary name, so users can manually recover data.
		// (No automatic recovery is possible because after crash the
		// temporary name is not known.)
		var origtmp string
		for {
			origtmp, err = makeTempName(newname, filepath.Base(newname))
			if err != nil {
				return err
			}
			_, err = os.Stat(origtmp)
			if err == nil {
				continue // most likely will never happen
			}
			break
		}
		err = os.Rename(newname, origtmp)
		if err != nil {
			return err
		}
		err = os.Rename(oldname, newname)
		if err != nil {
			// Rename still fails, try to revert original rename,
			// ignoring errors.
			os.Rename(origtmp, newname)
			return err
		}
		// Rename succeeded, now delete original file.
		os.Remove(origtmp)
	}
	return nil
}
