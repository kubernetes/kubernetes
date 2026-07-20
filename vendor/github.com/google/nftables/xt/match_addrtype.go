package xt

import (
	"github.com/google/nftables/alignedbuff"
)

// Rev. 0, see https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/xt_addrtype.h#L38
type AddrType struct {
	Source       uint16
	Dest         uint16
	InvertSource bool
	InvertDest   bool
}

type AddrTypeFlags uint32

const (
	AddrTypeUnspec AddrTypeFlags = 1 << iota
	AddrTypeUnicast
	AddrTypeLocal
	AddrTypeBroadcast
	AddrTypeAnycast
	AddrTypeMulticast
	AddrTypeBlackhole
	AddrTypeUnreachable
	AddrTypeProhibit
	AddrTypeThrow
	AddrTypeNat
	AddrTypeXresolve
)

// See https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/xt_addrtype.h#L31
type AddrTypeV1 struct {
	Source uint16
	Dest   uint16
	Flags  AddrTypeFlags
}

func (x *AddrType) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	ab := alignedbuff.New()
	ab.PutUint16(x.Source)
	ab.PutUint16(x.Dest)
	putBool32(&ab, x.InvertSource)
	putBool32(&ab, x.InvertDest)
	return ab.Data(), nil
}

func (x *AddrType) unmarshal(fam TableFamily, rev uint32, data []byte) error {
	ab := alignedbuff.NewWithData(data)
	var err error
	if x.Source, err = ab.Uint16(); err != nil {
		return nil
	}
	if x.Dest, err = ab.Uint16(); err != nil {
		return nil
	}
	if x.InvertSource, err = bool32(&ab); err != nil {
		return nil
	}
	if x.InvertDest, err = bool32(&ab); err != nil {
		return nil
	}
	return nil
}

func (x *AddrTypeV1) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	ab := alignedbuff.New()
	ab.PutUint16(x.Source)
	ab.PutUint16(x.Dest)
	ab.PutUint32(uint32(x.Flags))
	return ab.Data(), nil
}

func (x *AddrTypeV1) unmarshal(fam TableFamily, rev uint32, data []byte) error {
	ab := alignedbuff.NewWithData(data)
	var err error
	if x.Source, err = ab.Uint16(); err != nil {
		return nil
	}
	if x.Dest, err = ab.Uint16(); err != nil {
		return nil
	}
	var flags uint32
	if flags, err = ab.Uint32(); err != nil {
		return nil
	}
	x.Flags = AddrTypeFlags(flags)
	return nil
}
