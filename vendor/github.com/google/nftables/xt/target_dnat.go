package xt

import (
	"net"

	"github.com/google/nftables/alignedbuff"
)

type NatRangeFlags uint

// See: https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/nf_nat.h#L8
const (
	NatRangeMapIPs NatRangeFlags = (1 << iota)
	NatRangeProtoSpecified
	NatRangeProtoRandom
	NatRangePersistent
	NatRangeProtoRandomFully
	NatRangeProtoOffset
	NatRangeNetmap

	NatRangeMask NatRangeFlags = (1 << iota) - 1

	NatRangeProtoRandomAll = NatRangeProtoRandom | NatRangeProtoRandomFully
)

// see: https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/nf_nat.h#L38
type NatRange struct {
	Flags   uint   // sic! platform/arch/compiler-dependent uint size
	MinIP   net.IP // always taking up space for an IPv6 address
	MaxIP   net.IP // dito
	MinPort uint16
	MaxPort uint16
}

// see: https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/nf_nat.h#L46
type NatRange2 struct {
	NatRange
	BasePort uint16
}

func (x *NatRange) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	ab := alignedbuff.New()
	if err := x.marshalAB(fam, rev, &ab); err != nil {
		return nil, err
	}
	return ab.Data(), nil
}

func (x *NatRange) marshalAB(fam TableFamily, rev uint32, ab *alignedbuff.AlignedBuff) error {
	ab.PutUint(x.Flags)
	if err := putIPv46(ab, fam, x.MinIP); err != nil {
		return err
	}
	if err := putIPv46(ab, fam, x.MaxIP); err != nil {
		return err
	}
	ab.PutUint16BE(x.MinPort)
	ab.PutUint16BE(x.MaxPort)
	return nil
}

func (x *NatRange) unmarshal(fam TableFamily, rev uint32, data []byte) error {
	ab := alignedbuff.NewWithData(data)
	return x.unmarshalAB(fam, rev, &ab)
}

func (x *NatRange) unmarshalAB(fam TableFamily, rev uint32, ab *alignedbuff.AlignedBuff) error {
	var err error
	if x.Flags, err = ab.Uint(); err != nil {
		return err
	}
	if x.MinIP, err = iPv46(ab, fam); err != nil {
		return err
	}
	if x.MaxIP, err = iPv46(ab, fam); err != nil {
		return err
	}
	if x.MinPort, err = ab.Uint16BE(); err != nil {
		return err
	}
	if x.MaxPort, err = ab.Uint16BE(); err != nil {
		return err
	}
	return nil
}

func (x *NatRange2) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	ab := alignedbuff.New()
	if err := x.NatRange.marshalAB(fam, rev, &ab); err != nil {
		return nil, err
	}
	ab.PutUint16BE(x.BasePort)
	return ab.Data(), nil
}

func (x *NatRange2) unmarshal(fam TableFamily, rev uint32, data []byte) error {
	ab := alignedbuff.NewWithData(data)
	var err error
	if err = x.NatRange.unmarshalAB(fam, rev, &ab); err != nil {
		return err
	}
	if x.BasePort, err = ab.Uint16BE(); err != nil {
		return err
	}
	return nil
}
