// +build go1.7

package sdkio

import "io"

// Alias for Go 1.7 io package Seeker constants
const (
	SeekStart   = io.SeekStart   // seek relative to the origin of the file
	SeekCurrent = io.SeekCurrent // seek relative to the current offset
	SeekEnd     = io.SeekEnd     // seek relative to the end
)
