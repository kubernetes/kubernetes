package xt

import (
	"github.com/google/nftables/alignedbuff"
)

// Tcp is the Match.Info payload for the tcp xtables extension
// (https://wiki.nftables.org/wiki-nftables/index.php/Supported_features_compared_to_xtables#tcp).
//
// See
// https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/xt_tcpudp.h#L25
type Udp struct {
	SrcPorts [2]uint16     // min, max source port range
	DstPorts [2]uint16     // min, max destination port range
	InvFlags UdpInvFlagset // Inverse flags
}

type UdpInvFlagset uint8

const (
	UdpInvSrcPorts UdpInvFlagset = 1 << iota
	UdpInvDestPorts
	UdpInvMask UdpInvFlagset = (1 << iota) - 1
)

func (x *Udp) marshal(fam TableFamily, rev uint32) ([]byte, error) {
	ab := alignedbuff.New()
	ab.PutUint16(x.SrcPorts[0])
	ab.PutUint16(x.SrcPorts[1])
	ab.PutUint16(x.DstPorts[0])
	ab.PutUint16(x.DstPorts[1])
	ab.PutUint8(byte(x.InvFlags))
	return ab.Data(), nil
}

func (x *Udp) unmarshal(fam TableFamily, rev uint32, data []byte) error {
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
	var invFlags uint8
	if invFlags, err = ab.Uint8(); err != nil {
		return err
	}
	x.InvFlags = UdpInvFlagset(invFlags)
	return nil
}
