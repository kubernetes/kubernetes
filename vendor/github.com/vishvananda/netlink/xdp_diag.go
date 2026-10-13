package netlink

import "github.com/vishvananda/netlink/nl"

const SOCK_ANY_COOKIE = uint64(nl.TCPDIAG_NOCOOKIE)<<32 + uint64(nl.TCPDIAG_NOCOOKIE)

// XDP diagnosis show flag constants to request particular information elements.
const (
	XDP_SHOW_INFO = 1 << iota
	XDP_SHOW_RING_CFG
	XDP_SHOW_UMEM
	XDP_SHOW_MEMINFO
	XDP_SHOW_STATS
)

// XDP diag element constants
const (
	XDP_DIAG_NONE                 = iota
	XDP_DIAG_INFO                 // when using XDP_SHOW_INFO
	XDP_DIAG_UID                  // when using XDP_SHOW_INFO
	XDP_DIAG_RX_RING              // when using XDP_SHOW_RING_CFG
	XDP_DIAG_TX_RING              // when using XDP_SHOW_RING_CFG
	XDP_DIAG_UMEM                 // when using XDP_SHOW_UMEM
	XDP_DIAG_UMEM_FILL_RING       // when using XDP_SHOW_UMEM
	XDP_DIAG_UMEM_COMPLETION_RING // when using XDP_SHOW_UMEM
	XDP_DIAG_MEMINFO              // when using XDP_SHOW_MEMINFO
	XDP_DIAG_STATS                // when using XDP_SHOW_STATS
)

// https://elixir.bootlin.com/linux/v6.2/source/include/uapi/linux/xdp_diag.h#L21
type XDPDiagInfoResp struct {
	XDPDiagMsg *XDPSocket
	XDPInfo    *XDPInfo
}
