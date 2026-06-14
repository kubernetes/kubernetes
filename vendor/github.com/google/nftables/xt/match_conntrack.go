package xt

import (
	"net"

	"github.com/google/nftables/alignedbuff"
)

type ConntrackFlags uint16

const (
	ConntrackState ConntrackFlags = 1 << iota
	ConntrackProto
	ConntrackOrigSrc
	ConntrackOrigDst
	ConntrackReplSrc
	ConntrackReplDst
	ConntrackStatus
	ConntrackExpires
	ConntrackOrigSrcPort
	ConntrackOrigDstPort
	ConntrackReplSrcPort
	ConntrackReplDstPrt
	ConntrackDirection
	ConntrackStateAlias
)

type ConntrackMtinfoBase struct {
	OrigSrcAddr net.IP
	OrigSrcMask net.IPMask
	OrigDstAddr net.IP
	OrigDstMask net.IPMask
	ReplSrcAddr net.IP
	ReplSrcMask net.IPMask
	ReplDstAddr net.IP
	ReplDstMask net.IPMask
	ExpiresMin  uint32
	ExpiresMax  uint32
	L4Proto     uint16
	OrigSrcPort uint16
	OrigDstPort uint16
	ReplSrcPort uint16
	ReplDstPort uint16
	MatchFlags  uint16
	InvertFlags uint16
}

// See https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/xt_conntrack.h#L38
type ConntrackMtinfo1 struct {
	ConntrackMtinfoBase
	StateMask  uint8
	StatusMask uint8
}

// See https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/xt_conntrack.h#L51
type ConntrackMtinfo2 struct {
	ConntrackMtinfoBase
	StateMask  uint16
	StatusMask uint16
}

// See https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/xt_conntrack.h#L64
type ConntrackMtinfo3 struct {
	ConntrackMtinfo2
	OrigSrcPortHigh uint16
	OrigDstPortHigh uint16
	ReplSrcPortHigh uint16
	ReplDstPortHigh uint16
}

func (x *ConntrackMtinfoBase) marshalAB(fam TableFamily, rev uint32, ab *alignedbuff.AlignedBuff) error {
	if err := putIPv46(ab, fam, x.OrigSrcAddr); err != nil {
		return err
	}
	if err := putIPv46Mask(ab, fam, x.OrigSrcMask); err != nil {
		return err
	}
	if err := putIPv46(ab, fam, x.OrigDstAddr); err != nil {
		return err
	}
	if err := putIPv46Mask(ab, fam, x.OrigDstMask); err != nil {
		return err
	}
	if err := putIPv46(ab, fam, x.ReplSrcAddr); err != nil {
		return err
	}
	if err := putIPv46Mask(ab, fam, x.ReplSrcMask); err != nil {
		return err
	}
	if err := putIPv46(ab, fam, x.ReplDstAddr); err != nil {
		return err
	}
	if err := putIPv46Mask(ab, fam, x.ReplDstMask); err != nil {
		return err
	}
	ab.PutUint32(x.ExpiresMin)
	ab.PutUint32(x.ExpiresMax)
	ab.PutUint16(x.L4Proto)
	ab.PutUint16(x.OrigSrcPort)
	ab.PutUint16(x.OrigDstPort)
	ab.PutUint16(x.ReplSrcPort)
	ab.PutUint16(x.ReplDstPort)
	ab.PutUint16(x.MatchFlags)
	ab.PutUint16(x.InvertFlags)
	return nil
}

func (x *ConntrackMtinfoBase) unmarshalAB(fam TableFamily, rev uint32, ab *alignedbuff.AlignedBuff) error {
	var err error
	if x.OrigSrcAddr, err = iPv46(ab, fam); err != nil {
		return err
	}
	if x.OrigSrcMask, err = iPv46Mask(ab, fam); err != nil {
		return err
	}
	if x.OrigDstAddr, err = iPv46(ab, fam); err != nil {
		return err
	}
	if x.OrigDstMask, err = iPv46Mask(ab, fam); err != nil {
		return err
	}
	if x.ReplSrcAddr, err = iPv46(ab, fam); err != nil {
		return err
	}
	if x.ReplSrcMask, err = iPv46Mask(ab, fam); err != nil {
		return err
	}
	if x.ReplDstAddr, err = iPv46(ab, fam); err != nil {
		return err
	}
	if x.ReplDstMask, err = iPv46Mask(ab, fam); err != nil {
		return err
	}
	if x.ExpiresMin, err = ab.Uint32(); err != nil {
		return err
	}
	if x.ExpiresMax, err = ab.Uint32(); err != nil {
		return err
	}
	if x.L4Proto, err = ab.Uint16(); err != nil {
		return err
	}
	if x.OrigSrcPort, err = ab.Uint16(); err != nil {
		return err
	}
	if x.OrigDstPort, err = ab.Uint16(); err != nil {
		return err
	}
	if x.ReplSrcPort, err = ab.Uint16(); err != nil {
		return err
	}
	if x.ReplDstPort, err = ab.Uint16(); err != nil {
		return err
	}
	if x.MatchFlags, err = ab.Uint16(); err != nil {
		return err
	}
	if x.InvertFlags, err = ab.Uint16(); err != nil {
		return err
	}
	return nil
}

func (x *ConntrackMtinfo1) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	ab := alignedbuff.New()
	if err := x.ConntrackMtinfoBase.marshalAB(fam, rev, &ab); err != nil {
		return nil, err
	}
	ab.PutUint8(x.StateMask)
	ab.PutUint8(x.StatusMask)
	return ab.Data(), nil
}

func (x *ConntrackMtinfo1) unmarshal(fam TableFamily, rev uint32, data []byte) error {
	ab := alignedbuff.NewWithData(data)
	var err error
	if err = x.ConntrackMtinfoBase.unmarshalAB(fam, rev, &ab); err != nil {
		return err
	}
	if x.StateMask, err = ab.Uint8(); err != nil {
		return err
	}
	if x.StatusMask, err = ab.Uint8(); err != nil {
		return err
	}
	return nil
}

func (x *ConntrackMtinfo2) marshalAB(fam TableFamily, rev uint32, ab *alignedbuff.AlignedBuff) error {
	if err := x.ConntrackMtinfoBase.marshalAB(fam, rev, ab); err != nil {
		return err
	}
	ab.PutUint16(x.StateMask)
	ab.PutUint16(x.StatusMask)
	return nil
}

func (x *ConntrackMtinfo2) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	ab := alignedbuff.New()
	if err := x.marshalAB(fam, rev, &ab); err != nil {
		return nil, err
	}
	return ab.Data(), nil
}

func (x *ConntrackMtinfo2) unmarshalAB(fam TableFamily, rev uint32, ab *alignedbuff.AlignedBuff) error {
	var err error
	if err = x.ConntrackMtinfoBase.unmarshalAB(fam, rev, ab); err != nil {
		return err
	}
	if x.StateMask, err = ab.Uint16(); err != nil {
		return err
	}
	if x.StatusMask, err = ab.Uint16(); err != nil {
		return err
	}
	return nil
}

func (x *ConntrackMtinfo2) unmarshal(fam TableFamily, rev uint32, data []byte) error {
	ab := alignedbuff.NewWithData(data)
	var err error
	if err = x.unmarshalAB(fam, rev, &ab); err != nil {
		return err
	}
	return nil
}

func (x *ConntrackMtinfo3) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	ab := alignedbuff.New()
	if err := x.ConntrackMtinfo2.marshalAB(fam, rev, &ab); err != nil {
		return nil, err
	}
	ab.PutUint16(x.OrigSrcPortHigh)
	ab.PutUint16(x.OrigDstPortHigh)
	ab.PutUint16(x.ReplSrcPortHigh)
	ab.PutUint16(x.ReplDstPortHigh)
	return ab.Data(), nil
}

func (x *ConntrackMtinfo3) unmarshal(fam TableFamily, rev uint32, data []byte) error {
	ab := alignedbuff.NewWithData(data)
	var err error
	if err = x.ConntrackMtinfo2.unmarshalAB(fam, rev, &ab); err != nil {
		return err
	}
	if x.OrigSrcPortHigh, err = ab.Uint16(); err != nil {
		return err
	}
	if x.OrigDstPortHigh, err = ab.Uint16(); err != nil {
		return err
	}
	if x.ReplSrcPortHigh, err = ab.Uint16(); err != nil {
		return err
	}
	if x.ReplDstPortHigh, err = ab.Uint16(); err != nil {
		return err
	}
	return nil
}
