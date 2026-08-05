package xt

import (
	"github.com/google/nftables/alignedbuff"
)

// Tcp is the Match.Info payload for the tcp xtables extension
// (https://wiki.nftables.org/wiki-nftables/index.php/Supported_features_compared_to_xtables#tcp).
//
// See
// https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/xt_tcpudp.h#L8
type Tcp struct {
	SrcPorts  [2]uint16     // min, max source port range
	DstPorts  [2]uint16     // min, max destination port range
	Option    uint8         // TCP option if non-zero
	FlagsMask uint8         // TCP flags mask
	FlagsCmp  uint8         // TCP flags compare
	InvFlags  TcpInvFlagset // Inverse flags
}

type TcpInvFlagset uint8

const (
	TcpInvSrcPorts TcpInvFlagset = 1 << iota
	TcpInvDestPorts
	TcpInvFlags
	TcpInvOption
	TcpInvMask TcpInvFlagset = (1 << iota) - 1
)

func (x *Tcp) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	ab := alignedbuff.New()
	ab.PutUint16(x.SrcPorts[0])
	ab.PutUint16(x.SrcPorts[1])
	ab.PutUint16(x.DstPorts[0])
	ab.PutUint16(x.DstPorts[1])
	ab.PutUint8(x.Option)
	ab.PutUint8(x.FlagsMask)
	ab.PutUint8(x.FlagsCmp)
	ab.PutUint8(byte(x.InvFlags))
	return ab.Data(), nil
}

func (x *Tcp) unmarshal(fam TableFamily, rev uint32, data []byte) error {
	ab := alignedbuff.NewWithData(data)
	var err error
	if x.SrcPorts[0], err = ab.Uint16(); err != nil {
		return err
	}
	if x.SrcPorts[1], err = ab.Uint16(); err != nil {
		return err
	}
	if x.DstPorts[0], err = ab.Uint16(); err != nil {
		return err
	}
	if x.DstPorts[1], err = ab.Uint16(); err != nil {
		return err
	}
	if x.Option, err = ab.Uint8(); err != nil {
		return err
	}
	if x.FlagsMask, err = ab.Uint8(); err != nil {
		return err
	}
	if x.FlagsCmp, err = ab.Uint8(); err != nil {
		return err
	}
	var invFlags uint8
	if invFlags, err = ab.Uint8(); err != nil {
		return err
	}
	x.InvFlags = TcpInvFlagset(invFlags)
	return nil
}
