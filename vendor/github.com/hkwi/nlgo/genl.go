// +build linux

package nlgo

import (
	"syscall"
	"unsafe"
)

const (
	GENL_ADMIN_PERM = 1 << iota
	GENL_CMD_CAP_DO
	GENL_CMD_CAP_DUMP
	GENL_CMD_CAP_HASPOL
)

// GenlConnect is same with libnl genl_connect.
func GenlConnect(sk *NlSock) error {
	return NlConnect(sk, syscall.NETLINK_GENERIC)
}

// GenlSendSimple is same with libnl genl_send_simple.
func GenlSendSimple(sk *NlSock, family uint16, cmd, version uint8, flags uint16) error {
	hdr := (*[SizeofGenlMsghdr]byte)(unsafe.Pointer(&GenlMsghdr{
		Cmd:     cmd,
		Version: version,
	}))
	return NlSendSimple(sk, family, flags, hdr[:])
}

const (
	GENL_ID_GENERATE = 0
	GENL_ID_CTRL     = 0x10
)

const CTRL_VERSION = 0x0001

const (
	CTRL_CMD_UNSPEC = iota
	CTRL_CMD_NEWFAMILY
	CTRL_CMD_DELFAMILY
	CTRL_CMD_GETFAMILY
	CTRL_CMD_NEWOPS
	CTRL_CMD_DELOPS
	CTRL_CMD_GETOPS
	CTRL_CMD_NEWMCAST_GRP
	CTRL_CMD_DELMCAST_GRP
	CTRL_CMD_GETMCAST_GRP
)

// CTRL

const (
	CTRL_ATTR_UNSPEC = iota
	CTRL_ATTR_FAMILY_ID
	CTRL_ATTR_FAMILY_NAME
	CTRL_ATTR_VERSION
	CTRL_ATTR_HDRSIZE
	CTRL_ATTR_MAXATTR
	CTRL_ATTR_OPS
	CTRL_ATTR_MCAST_GROUPS
)

const (
	CTRL_ATTR_OP_UNSPEC = iota
	CTRL_ATTR_OP_ID
	CTRL_ATTR_OP_FLAGS // GENL_CMD_CAP_DUMP, etc.,
)

const (
	CTRL_ATTR_MCAST_GRP_UNSPEC = iota
	CTRL_ATTR_MCAST_GRP_NAME
	CTRL_ATTR_MCAST_GRP_ID
)

var CtrlPolicy MapPolicy = MapPolicy{
	Prefix: "CTRL_ATTR",
	Names:  CTRL_ATTR_itoa,
	Rule: map[uint16]Policy{
		CTRL_ATTR_FAMILY_ID:   U16Policy,
		CTRL_ATTR_FAMILY_NAME: NulStringPolicy,
		CTRL_ATTR_VERSION:     U32Policy,
		CTRL_ATTR_HDRSIZE:     U32Policy,
		CTRL_ATTR_MAXATTR:     U32Policy,
		CTRL_ATTR_OPS: ListPolicy{
			Nested: MapPolicy{
				Prefix: "OP",
				Names:  CTRL_ATTR_OP_itoa,
				Rule: map[uint16]Policy{
					CTRL_ATTR_OP_ID:    U32Policy,
					CTRL_ATTR_OP_FLAGS: U32Policy,
				},
			},
		},
		CTRL_ATTR_MCAST_GROUPS: ListPolicy{
			Nested: MapPolicy{
				Prefix: "MCAST_GRP",
				Names:  CTRL_ATTR_MCAST_GRP_itoa,
				Rule: map[uint16]Policy{
					CTRL_ATTR_MCAST_GRP_NAME: NulStringPolicy,
					CTRL_ATTR_MCAST_GRP_ID:   U32Policy,
				},
			},
		},
	},
}
