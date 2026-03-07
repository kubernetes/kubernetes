package xt

import (
	"errors"
	"net"

	"github.com/google/nftables/alignedbuff"
)

// See https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/nf_nat.h#L25
type NatIPv4Range struct {
	Flags   uint // sic!
	MinIP   net.IP
	MaxIP   net.IP
	MinPort uint16
	MaxPort uint16
}

// NatIPv4MultiRangeCompat despite being a slice of NAT IPv4 ranges is currently allowed to
// only hold exactly one element.
//
// See https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/nf_nat.h#L33
type NatIPv4MultiRangeCompat []NatIPv4Range

func (x *NatIPv4MultiRangeCompat) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	ab := alignedbuff.New()
	if len(*x) != 1 {
		return nil, errors.New("MasqueradeIp must contain exactly one NatIPv4Range")
	}
	ab.PutUint(uint(len(*x)))
	for _, nat := range *x {
		if err := nat.marshalAB(fam, rev, &ab); err != nil {
			return nil, err
		}
	}
	return ab.Data(), nil
}

func (x *NatIPv4MultiRangeCompat) unmarshal(fam TableFamily, rev uint32, data []byte) error {
	ab := alignedbuff.NewWithData(data)
	l, err := ab.Uint()
	if err != nil {
		return err
	}
	nats := make(NatIPv4MultiRangeCompat, l)
	for l > 0 {
		l--
		if err := nats[l].unmarshalAB(fam, rev, &ab); err != nil {
			return err
		}
	}
	*x = nats
	return nil
}

func (x *NatIPv4Range) marshalAB(fam TableFamily, rev uint32, ab *alignedbuff.AlignedBuff) error {
	ab.PutUint(x.Flags)
	ab.PutBytesAligned32(x.MinIP.To4(), 4)
	ab.PutBytesAligned32(x.MaxIP.To4(), 4)
	ab.PutUint16BE(x.MinPort)
	ab.PutUint16BE(x.MaxPort)
	return nil
}

func (x *NatIPv4Range) unmarshalAB(fam TableFamily, rev uint32, ab *alignedbuff.AlignedBuff) error {
	var err error
	if x.Flags, err = ab.Uint(); err != nil {
		return err
	}
	var ip []byte
	if ip, err = ab.BytesAligned32(4); err != nil {
		return err
	}
	x.MinIP = net.IP(ip)
	if ip, err = ab.BytesAligned32(4); err != nil {
		return err
	}
	x.MaxIP = net.IP(ip)
	if x.MinPort, err = ab.Uint16BE(); err != nil {
		return err
	}
	if x.MaxPort, err = ab.Uint16BE(); err != nil {
		return err
	}
	return nil
}
