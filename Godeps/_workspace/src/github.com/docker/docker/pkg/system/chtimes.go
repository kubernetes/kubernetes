package system

import (
	"os"
	"time"
)

// Chtimes changes the access time and modified time of a file at the given path
func Chtimes(name string, atime time.Time, mtime time.Time) error {
	unixMinTime := time.Unix(0, 0)
	// The max Unix time is 33 bits set
	unixMaxTime := unixMinTime.Add((1<<33 - 1) * time.Second)

	// If the modified time is prior to the Unix Epoch, or after the
	// end of Unix Time, os.Chtimes has undefined behavior
	// default to Unix Epoch in this case, just in case

	if atime.Before(unixMinTime) || atime.After(unixMaxTime) {
		atime = unixMinTime
	}

	if mtime.Before(unixMinTime) || mtime.After(unixMaxTime) {
		mtime = unixMinTime
	}

	if err := os.Chtimes(name, atime, mtime); err != nil {
		return err
	}

	return nil
}
