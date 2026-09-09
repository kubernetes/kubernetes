package xt

import (
	"golang.org/x/sys/unix"
)

// TableFamily specifies the address family of the table Match or Target Info
// data is contained in. On purpose, we don't import the expr package here in
// order to keep the option open to import this package instead into expr.
type TableFamily byte

// InfoAny is a (un)marshaling implemented by any info type.
type InfoAny interface {
	marshal(fam TableFamily, rev uint32) ([]byte, error)
	unmarshal(fam TableFamily, rev uint32, data []byte) error
}

// Marshal a Match or Target Info type into its binary representation.
func Marshal(fam TableFamily, rev uint32, info InfoAny) ([]byte, error) {
	return info.marshal(fam, rev)
}

// Unmarshal Info binary payload into its corresponding dedicated type as
// indicated by the name argument. In several cases, unmarshalling depends on
// the specific table family the Target or Match expression with the info
// payload belongs to, as well as the specific info structure revision.
func Unmarshal(name string, fam TableFamily, rev uint32, data []byte) (InfoAny, error) {
	var i InfoAny
	switch name {
	case "addrtype":
		switch rev {
		case 0:
			i = &AddrType{}
		case 1:
			i = &AddrTypeV1{}
		}
	case "comment":
		var c Comment
		i = &c
	case "conntrack":
		switch rev {
		case 1:
			i = &ConntrackMtinfo1{}
		case 2:
			i = &ConntrackMtinfo2{}
		case 3:
			i = &ConntrackMtinfo3{}
		}
	case "tcp":
		i = &Tcp{}
	case "udp":
		i = &Udp{}
	case "SNAT":
		if fam == unix.NFPROTO_IPV4 {
			i = &NatIPv4MultiRangeCompat{}
		}
	case "DNAT":
		switch fam {
		case unix.NFPROTO_IPV4:
			if rev == 0 {
				i = &NatIPv4MultiRangeCompat{}
				break
			}
			fallthrough
		case unix.NFPROTO_IPV6:
			switch rev {
			case 1:
				i = &NatRange{}
			case 2:
				i = &NatRange2{}
			}
		}
	case "MASQUERADE":
		switch fam {
		case unix.NFPROTO_IPV4:
			i = &NatIPv4MultiRangeCompat{}
		}
	case "REDIRECT":
		switch fam {
		case unix.NFPROTO_IPV4:
			if rev == 0 {
				i = &NatIPv4MultiRangeCompat{}
				break
			}
			fallthrough
		case unix.NFPROTO_IPV6:
			i = &NatRange{}
		}
	}
	if i == nil {
		i = &Unknown{}
	}
	if err := i.unmarshal(fam, rev, data); err != nil {
		return nil, err
	}
	return i, nil
}
