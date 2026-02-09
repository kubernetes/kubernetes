package nl

import (
	"strconv"

	"golang.org/x/sys/unix"
)

const (
	/* The protocol version */
	IPSET_PROTOCOL = 6

	/* The max length of strings including NUL: set and type identifiers */
	IPSET_MAXNAMELEN = 32

	/* The maximum permissible comment length we will accept over netlink */
	IPSET_MAX_COMMENT_SIZE = 255
)

const (
	_                  = iota
	IPSET_CMD_PROTOCOL /* 1: Return protocol version */
	IPSET_CMD_CREATE   /* 2: Create a new (empty) set */
	IPSET_CMD_DESTROY  /* 3: Destroy a (empty) set */
	IPSET_CMD_FLUSH    /* 4: Remove all elements from a set */
	IPSET_CMD_RENAME   /* 5: Rename a set */
	IPSET_CMD_SWAP     /* 6: Swap two sets */
	IPSET_CMD_LIST     /* 7: List sets */
	IPSET_CMD_SAVE     /* 8: Save sets */
	IPSET_CMD_ADD      /* 9: Add an element to a set */
	IPSET_CMD_DEL      /* 10: Delete an element from a set */
	IPSET_CMD_TEST     /* 11: Test an element in a set */
	IPSET_CMD_HEADER   /* 12: Get set header data only */
	IPSET_CMD_TYPE     /* 13: Get set type */
)

/* Attributes at command level */
const (
	_                       = iota
	IPSET_ATTR_PROTOCOL     /* 1: Protocol version */
	IPSET_ATTR_SETNAME      /* 2: Name of the set */
	IPSET_ATTR_TYPENAME     /* 3: Typename */
	IPSET_ATTR_REVISION     /* 4: Settype revision */
	IPSET_ATTR_FAMILY       /* 5: Settype family */
	IPSET_ATTR_FLAGS        /* 6: Flags at command level */
	IPSET_ATTR_DATA         /* 7: Nested attributes */
	IPSET_ATTR_ADT          /* 8: Multiple data containers */
	IPSET_ATTR_LINENO       /* 9: Restore lineno */
	IPSET_ATTR_PROTOCOL_MIN /* 10: Minimal supported version number */

	IPSET_ATTR_SETNAME2     = IPSET_ATTR_TYPENAME     /* Setname at rename/swap */
	IPSET_ATTR_REVISION_MIN = IPSET_ATTR_PROTOCOL_MIN /* type rev min */
)

/* CADT specific attributes */
const (
	IPSET_ATTR_IP          = 1
	IPSET_ATTR_IP_FROM     = 1
	IPSET_ATTR_IP_TO       = 2
	IPSET_ATTR_CIDR        = 3
	IPSET_ATTR_PORT        = 4
	IPSET_ATTR_PORT_FROM   = 4
	IPSET_ATTR_PORT_TO     = 5
	IPSET_ATTR_TIMEOUT     = 6
	IPSET_ATTR_PROTO       = 7
	IPSET_ATTR_CADT_FLAGS  = 8
	IPSET_ATTR_CADT_LINENO = IPSET_ATTR_LINENO /* 9 */
	IPSET_ATTR_MARK        = 10
	IPSET_ATTR_MARKMASK    = 11

	/* Reserve empty slots */
	IPSET_ATTR_CADT_MAX = 16

	/* Create-only specific attributes */
	IPSET_ATTR_GC = 3 + iota
	IPSET_ATTR_HASHSIZE
	IPSET_ATTR_MAXELEM
	IPSET_ATTR_NETMASK
	IPSET_ATTR_PROBES
	IPSET_ATTR_RESIZE
	IPSET_ATTR_SIZE

	/* Kernel-only */
	IPSET_ATTR_ELEMENTS
	IPSET_ATTR_REFERENCES
	IPSET_ATTR_MEMSIZE

	SET_ATTR_CREATE_MAX
)

const (
	IPSET_ATTR_IPADDR_IPV4 = 1
	IPSET_ATTR_IPADDR_IPV6 = 2
)

/* ADT specific attributes */
const (
	IPSET_ATTR_ETHER = IPSET_ATTR_CADT_MAX + iota + 1
	IPSET_ATTR_NAME
	IPSET_ATTR_NAMEREF
	IPSET_ATTR_IP2
	IPSET_ATTR_CIDR2
	IPSET_ATTR_IP2_TO
	IPSET_ATTR_IFACE
	IPSET_ATTR_BYTES
	IPSET_ATTR_PACKETS
	IPSET_ATTR_COMMENT
	IPSET_ATTR_SKBMARK
	IPSET_ATTR_SKBPRIO
	IPSET_ATTR_SKBQUEUE
)

/* Flags at CADT attribute level, upper half of cmdattrs */
const (
	IPSET_FLAG_BIT_BEFORE        = 0
	IPSET_FLAG_BEFORE            = (1 << IPSET_FLAG_BIT_BEFORE)
	IPSET_FLAG_BIT_PHYSDEV       = 1
	IPSET_FLAG_PHYSDEV           = (1 << IPSET_FLAG_BIT_PHYSDEV)
	IPSET_FLAG_BIT_NOMATCH       = 2
	IPSET_FLAG_NOMATCH           = (1 << IPSET_FLAG_BIT_NOMATCH)
	IPSET_FLAG_BIT_WITH_COUNTERS = 3
	IPSET_FLAG_WITH_COUNTERS     = (1 << IPSET_FLAG_BIT_WITH_COUNTERS)
	IPSET_FLAG_BIT_WITH_COMMENT  = 4
	IPSET_FLAG_WITH_COMMENT      = (1 << IPSET_FLAG_BIT_WITH_COMMENT)
	IPSET_FLAG_BIT_WITH_FORCEADD = 5
	IPSET_FLAG_WITH_FORCEADD     = (1 << IPSET_FLAG_BIT_WITH_FORCEADD)
	IPSET_FLAG_BIT_WITH_SKBINFO  = 6
	IPSET_FLAG_WITH_SKBINFO      = (1 << IPSET_FLAG_BIT_WITH_SKBINFO)
	IPSET_FLAG_CADT_MAX          = 15
)

const (
	IPSET_ERR_PRIVATE = 4096 + iota
	IPSET_ERR_PROTOCOL
	IPSET_ERR_FIND_TYPE
	IPSET_ERR_MAX_SETS
	IPSET_ERR_BUSY
	IPSET_ERR_EXIST_SETNAME2
	IPSET_ERR_TYPE_MISMATCH
	IPSET_ERR_EXIST
	IPSET_ERR_INVALID_CIDR
	IPSET_ERR_INVALID_NETMASK
	IPSET_ERR_INVALID_FAMILY
	IPSET_ERR_TIMEOUT
	IPSET_ERR_REFERENCED
	IPSET_ERR_IPADDR_IPV4
	IPSET_ERR_IPADDR_IPV6
	IPSET_ERR_COUNTER
	IPSET_ERR_COMMENT
	IPSET_ERR_INVALID_MARKMASK
	IPSET_ERR_SKBINFO

	/* Type specific error codes */
	IPSET_ERR_TYPE_SPECIFIC = 4352
)

type IPSetError uintptr

func (e IPSetError) Error() string {
	switch int(e) {
	case IPSET_ERR_PRIVATE:
		return "private"
	case IPSET_ERR_PROTOCOL:
		return "invalid protocol"
	case IPSET_ERR_FIND_TYPE:
		return "invalid type"
	case IPSET_ERR_MAX_SETS:
		return "max sets reached"
	case IPSET_ERR_BUSY:
		return "busy"
	case IPSET_ERR_EXIST_SETNAME2:
		return "exist_setname2"
	case IPSET_ERR_TYPE_MISMATCH:
		return "type mismatch"
	case IPSET_ERR_EXIST:
		return "exist"
	case IPSET_ERR_INVALID_CIDR:
		return "invalid cidr"
	case IPSET_ERR_INVALID_NETMASK:
		return "invalid netmask"
	case IPSET_ERR_INVALID_FAMILY:
		return "invalid family"
	case IPSET_ERR_TIMEOUT:
		return "timeout"
	case IPSET_ERR_REFERENCED:
		return "referenced"
	case IPSET_ERR_IPADDR_IPV4:
		return "invalid ipv4 address"
	case IPSET_ERR_IPADDR_IPV6:
		return "invalid ipv6 address"
	case IPSET_ERR_COUNTER:
		return "invalid counter"
	case IPSET_ERR_COMMENT:
		return "invalid comment"
	case IPSET_ERR_INVALID_MARKMASK:
		return "invalid markmask"
	case IPSET_ERR_SKBINFO:
		return "skbinfo"
	default:
		return "errno " + strconv.Itoa(int(e))
	}
}

func GetIpsetFlags(cmd int) int {
	switch cmd {
	case IPSET_CMD_CREATE:
		return unix.NLM_F_REQUEST | unix.NLM_F_ACK | unix.NLM_F_CREATE
	case IPSET_CMD_DESTROY,
		IPSET_CMD_FLUSH,
		IPSET_CMD_RENAME,
		IPSET_CMD_SWAP,
		IPSET_CMD_TEST:
		return unix.NLM_F_REQUEST | unix.NLM_F_ACK
	case IPSET_CMD_LIST,
		IPSET_CMD_SAVE:
		return unix.NLM_F_REQUEST | unix.NLM_F_ACK | unix.NLM_F_ROOT | unix.NLM_F_MATCH | unix.NLM_F_DUMP
	case IPSET_CMD_ADD,
		IPSET_CMD_DEL:
		return unix.NLM_F_REQUEST | unix.NLM_F_ACK
	case IPSET_CMD_HEADER,
		IPSET_CMD_TYPE,
		IPSET_CMD_PROTOCOL:
		return unix.NLM_F_REQUEST
	default:
		return 0
	}
}
