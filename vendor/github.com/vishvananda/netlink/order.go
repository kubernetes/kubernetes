package netlink

import (
	"encoding/binary"

	"github.com/vishvananda/netlink/nl"
)

var (
	native       = nl.NativeEndian()
	networkOrder = binary.BigEndian
)

func htonl(val uint32) []byte {
	bytes := make([]byte, 4)
	binary.BigEndian.PutUint32(bytes, val)
	return bytes
}

func htons(val uint16) []byte {
	bytes := make([]byte, 2)
	binary.BigEndian.PutUint16(bytes, val)
	return bytes
}

func ntohl(buf []byte) uint32 {
	return binary.BigEndian.Uint32(buf)
}

func ntohs(buf []byte) uint16 {
	return binary.BigEndian.Uint16(buf)
}
