package dmz

import (
	"errors"
)

// ErrNoDmzBinary is returned by Binary when there is no runc-dmz binary
// embedded in the runc program.
var ErrNoDmzBinary = errors.New("runc-dmz binary not embedded in this program")
