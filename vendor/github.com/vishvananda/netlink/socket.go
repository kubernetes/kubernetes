package netlink

import "net"

// SocketID identifies a single socket.
type SocketID struct {
	SourcePort      uint16
	DestinationPort uint16
	Source          net.IP
	Destination     net.IP
	Interface       uint32
	Cookie          [2]uint32
}

// Socket represents a netlink socket.
type Socket struct {
	Family  uint8
	State   uint8
	Timer   uint8
	Retrans uint8
	ID      SocketID
	Expires uint32
	RQueue  uint32
	WQueue  uint32
	UID     uint32
	INode   uint32
}

// UnixSocket represents a netlink unix socket.
type UnixSocket struct {
	Type   uint8
	Family uint8
	State  uint8
	pad    uint8
	INode  uint32
	Cookie [2]uint32
}

// XDPSocket represents an XDP socket (and the common diagnosis part in
// particular). Please note that in contrast to [UnixSocket] the XDPSocket type
// does not feature “State” information.
type XDPSocket struct {
	// xdp_diag_msg
	// https://elixir.bootlin.com/linux/v6.2/source/include/uapi/linux/xdp_diag.h#L21
	Family uint8
	Type   uint8
	pad    uint16
	Ino    uint32
	Cookie [2]uint32
}

type XDPInfo struct {
	// XDP_DIAG_INFO/xdp_diag_info
	// https://elixir.bootlin.com/linux/v6.2/source/include/uapi/linux/xdp_diag.h#L51
	Ifindex uint32
	QueueID uint32

	// XDP_DIAG_UID
	UID uint32

	// XDP_RX_RING
	// https://elixir.bootlin.com/linux/v6.2/source/include/uapi/linux/xdp_diag.h#L56
	RxRingEntries             uint32
	TxRingEntries             uint32
	UmemFillRingEntries       uint32
	UmemCompletionRingEntries uint32

	// XDR_DIAG_UMEM
	Umem *XDPDiagUmem

	// XDR_DIAG_STATS
	Stats *XDPDiagStats
}

const (
	XDP_DU_F_ZEROCOPY = 1 << iota
)

// XDPDiagUmem describes the umem attached to an XDP socket.
//
// https://elixir.bootlin.com/linux/v6.2/source/include/uapi/linux/xdp_diag.h#L62
type XDPDiagUmem struct {
	Size      uint64
	ID        uint32
	NumPages  uint32
	ChunkSize uint32
	Headroom  uint32
	Ifindex   uint32
	QueueID   uint32
	Flags     uint32
	Refs      uint32
}

// XDPDiagStats contains ring statistics for an XDP socket.
//
// https://elixir.bootlin.com/linux/v6.2/source/include/uapi/linux/xdp_diag.h#L74
type XDPDiagStats struct {
	RxDropped     uint64
	RxInvalid     uint64
	RxFull        uint64
	FillRingEmpty uint64
	TxInvalid     uint64
	TxRingEmpty   uint64
}
