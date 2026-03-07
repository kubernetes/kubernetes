// Package nlenc implements encoding and decoding functions for netlink
// messages and attributes.
package nlenc

import (
	"encoding/binary"
)

// NativeEndian returns the native byte order of this system.
func NativeEndian() binary.ByteOrder {
	// TODO(mdlayher): consider deprecating and removing this function for v2.
	return binary.NativeEndian
}
