package bolt

import (
	"syscall"
)

var odirect = syscall.O_DIRECT

// fdatasync flushes written data to a file descriptor.
func fdatasync(db *DB) error {
	return syscall.Fdatasync(int(db.file.Fd()))
}
