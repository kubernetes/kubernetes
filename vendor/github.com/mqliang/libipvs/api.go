package libipvs

import (
	"fmt"
	"net"
	"syscall"

	"github.com/hkwi/nlgo"
)

type AddressFamily uint16

func (af AddressFamily) String() string {
	switch af {
	case syscall.AF_INET:
		return "inet"
	case syscall.AF_INET6:
		return "inet6"
	default:
		return fmt.Sprintf("%d", af)
	}
}

type Protocol uint16

func (p Protocol) String() string {
	switch p {
	case syscall.IPPROTO_TCP:
		return "tcp"
	case syscall.IPPROTO_UDP:
		return "udp"
	case syscall.IPPROTO_SCTP:
		return "sctp"
	default:
		return fmt.Sprintf("%d", p)
	}
}

type Flags struct {
	Flags uint32
	Mask  uint32
}

// Service defines an IPVS service in its entirety.
type Service struct {
	// Virtual service address.
	Address  net.IP
	Protocol Protocol
	Port     uint16
	FWMark   uint32 // Firewall mark of the service.

	// Virtual service options.
	SchedName     string
	Flags         Flags
	Timeout       uint32
	Netmask       uint32
	AddressFamily AddressFamily
	PEName        string
}

// Destination defines an IPVS destination (real server) in its
// entirety.
type Destination struct {
	Address   net.IP
	Port      uint16

	FwdMethod FwdMethod
	Weight    uint32

	UThresh uint32
	LThresh uint32

	ActiveConns  uint32
	InactConns   uint32
	PersistConns uint32

	AddressFamily AddressFamily
}

// Pack Service to a set of nlattrs.
// If full is given, include service settings, otherwise only the identifying fields are given.
func (self *Service) attrs(full bool) nlgo.AttrSlice {
	var attrs nlgo.AttrSlice
	if self.FWMark != 0 {
		attrs = append(attrs,
			nlattr(IPVS_SVC_ATTR_AF, nlgo.U16(self.AddressFamily)),
			nlattr(IPVS_SVC_ATTR_FWMARK, nlgo.U32(self.FWMark)),
		)
	} else if self.Protocol != 0 && self.Address != nil && self.Port != 0 {
		attrs = append(attrs,
			nlattr(IPVS_SVC_ATTR_AF, nlgo.U16(self.AddressFamily)),
			nlattr(IPVS_SVC_ATTR_PROTOCOL, nlgo.U16(self.Protocol)),
			nlattr(IPVS_SVC_ATTR_ADDR, packAddr(self.AddressFamily, self.Address)),
			nlattr(IPVS_SVC_ATTR_PORT, packPort(self.Port)),
		)
	} else {
		panic("Incomplete service id fields")
	}

	if full {
		attrs = append(attrs,
			nlattr(IPVS_SVC_ATTR_SCHED_NAME, nlgo.NulString(self.SchedName)),
			nlattr(IPVS_SVC_ATTR_FLAGS, pack(&self.Flags)),
			nlattr(IPVS_SVC_ATTR_TIMEOUT, nlgo.U32(self.Timeout)),
			nlattr(IPVS_SVC_ATTR_NETMASK, nlgo.U32(self.Netmask)),
		)
	}

	return attrs
}

// Dump Dest as nl attrs, using the Af of the corresponding Service.
// If full, includes Dest setting attrs, otherwise only identifying attrs.
func (self *Destination) attrs(service *Service, full bool) nlgo.AttrSlice {
	var attrs nlgo.AttrSlice

	attrs = append(attrs,
		nlattr(IPVS_DEST_ATTR_ADDR, packAddr(self.AddressFamily, self.Address)),
		nlattr(IPVS_DEST_ATTR_PORT, packPort(self.Port)),
	)

	if full {
		attrs = append(attrs,
			nlattr(IPVS_DEST_ATTR_FWD_METHOD, nlgo.U32(self.FwdMethod)),
			nlattr(IPVS_DEST_ATTR_WEIGHT, nlgo.U32(self.Weight)),
			nlattr(IPVS_DEST_ATTR_U_THRESH, nlgo.U32(self.UThresh)),
			nlattr(IPVS_DEST_ATTR_L_THRESH, nlgo.U32(self.LThresh)),
		)
	}

	return attrs
}

type FwdMethod uint32

func (self FwdMethod) String() string {
	switch value := (uint32(self) & IP_VS_CONN_F_FWD_MASK); value {
	case IP_VS_CONN_F_MASQ:
		return "masq"
	case IP_VS_CONN_F_LOCALNODE:
		return "localnode"
	case IP_VS_CONN_F_TUNNEL:
		return "tunnel"
	case IP_VS_CONN_F_DROUTE:
		return "droute"
	case IP_VS_CONN_F_BYPASS:
		return "bypass"
	default:
		return fmt.Sprintf("%#04x", value)
	}
}

func ParseFwdMethod(value string) (FwdMethod, error) {
	switch value {
	case "masq":
		return IP_VS_CONN_F_MASQ, nil
	case "tunnel":
		return IP_VS_CONN_F_TUNNEL, nil
	case "droute":
		return IP_VS_CONN_F_DROUTE, nil
	default:
		return 0, fmt.Errorf("Invalid FwdMethod: %s", value)
	}
}

type Version uint32

func (version Version) String() string {
	return fmt.Sprintf("%d.%d.%d",
		(version>>16)&0xFF,
		(version>>8)&0xFF,
		(version>>0)&0xFF,
	)
}

type Info struct {
	Version     Version
	ConnTabSize uint32
}
