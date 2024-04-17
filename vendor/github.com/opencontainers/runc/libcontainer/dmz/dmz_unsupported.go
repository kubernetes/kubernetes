//go:build !linux || runc_nodmz
// +build !linux runc_nodmz

package dmz

import (
	"os"
)

func Binary(_ string) (*os.File, error) {
	return nil, ErrNoDmzBinary
}
