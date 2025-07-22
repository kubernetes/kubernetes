package bbolt

import (
	"golang.org/x/sys/unix"
)

func msync(db *DB) error {
	return unix.Msync(db.data[:db.datasz], unix.MS_INVALIDATE)
}

func fdatasync(db *DB) error {
	if db.data != nil {
		return msync(db)
	}
	return db.file.Sync()
}
