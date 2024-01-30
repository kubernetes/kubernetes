//go:build !windows
// +build !windows

package bbolt

import "golang.org/x/sys/unix"

// mlock locks memory of db file
func mlock(db *DB, fileSize int) error {
	sizeToLock := fileSize
	if sizeToLock > db.datasz {
		// Can't lock more than mmaped slice
		sizeToLock = db.datasz
	}
	if err := unix.Mlock(db.dataref[:sizeToLock]); err != nil {
		return err
	}
	return nil
}

// munlock unlocks memory of db file
func munlock(db *DB, fileSize int) error {
	if db.dataref == nil {
		return nil
	}

	sizeToUnlock := fileSize
	if sizeToUnlock > db.datasz {
		// Can't unlock more than mmaped slice
		sizeToUnlock = db.datasz
	}

	if err := unix.Munlock(db.dataref[:sizeToUnlock]); err != nil {
		return err
	}
	return nil
}
