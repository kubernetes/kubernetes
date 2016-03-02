package zfs

import (
	"fmt"
)

// Error is an error which is returned when the `zfs` or `zpool` shell
// commands return with a non-zero exit code.
type Error struct {
	Err    error
	Debug  string
	Stderr string
}

// Error returns the string representation of an Error.
func (e Error) Error() string {
	return fmt.Sprintf("%s: %q => %s", e.Err, e.Debug, e.Stderr)
}
