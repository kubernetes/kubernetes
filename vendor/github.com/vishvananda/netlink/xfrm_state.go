package netlink

import (
	"net"
)

// XfrmStateAlgo represents the algorithm to use for the ipsec encryption.
type XfrmStateAlgo struct {
	Name        string
	Key         []byte
	TruncateLen int // Auth only
}

// EncapType is an enum representing an ipsec template direction.
type EncapType uint8

const (
	XFRM_ENCAP_ESPINUDP_NONIKE EncapType = iota + 1
	XFRM_ENCAP_ESPINUDP
)

func (e EncapType) String() string {
	switch e {
	case XFRM_ENCAP_ESPINUDP_NONIKE:
		return "espinudp-nonike"
	case XFRM_ENCAP_ESPINUDP:
		return "espinudp"
	}
	return "unknown"
}

// XfrmEncap represents the encapsulation to use for the ipsec encryption.
type XfrmStateEncap struct {
	Type            EncapType
	SrcPort         int
	DstPort         int
	OriginalAddress net.IP
}

// XfrmState represents the state of an ipsec policy. It optionally
// contains an XfrmStateAlgo for encryption and one for authentication.
type XfrmState struct {
	Dst          net.IP
	Src          net.IP
	Proto        Proto
	Mode         Mode
	Spi          int
	Reqid        int
	ReplayWindow int
	Auth         *XfrmStateAlgo
	Crypt        *XfrmStateAlgo
	Encap        *XfrmStateEncap
}
