// +build !windows,!plan9,!linux,!openbsd

package bolt

// fdatasync flushes written data to a file descriptor.
func fdatasync(db *DB) error {
	return db.file.Sync()
}
