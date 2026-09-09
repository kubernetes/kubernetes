package xt

import (
	"fmt"
	"net"

	"github.com/google/nftables/alignedbuff"
	"golang.org/x/sys/unix"
)

func bool32(ab *alignedbuff.AlignedBuff) (bool, error) {
	v, err := ab.Uint32()
	if err != nil {
		return false, err
	}
	if v != 0 {
		return true, nil
	}
	return false, nil
}

func putBool32(ab *alignedbuff.AlignedBuff, b bool) {
	if b {
		ab.PutUint32(1)
		return
	}
	ab.PutUint32(0)
}

func iPv46(ab *alignedbuff.AlignedBuff, fam TableFamily) (net.IP, error) {
	ip, err := ab.BytesAligned32(16)
	if err != nil {
		return nil, err
	}
	switch fam {
	case unix.NFPROTO_IPV4:
		return net.IP(ip[:4]), nil
	case unix.NFPROTO_IPV6:
		return net.IP(ip), nil
	default:
		return nil, fmt.Errorf("unmarshal IP: unsupported table family %d", fam)
	}
}

func iPv46Mask(ab *alignedbuff.AlignedBuff, fam TableFamily) (net.IPMask, error) {
	v, err := iPv46(ab, fam)
	return net.IPMask(v), err
}

func putIPv46(ab *alignedbuff.AlignedBuff, fam TableFamily, ip net.IP) error {
	switch fam {
	case unix.NFPROTO_IPV4:
		ab.PutBytesAligned32(ip.To4(), 16)
	case unix.NFPROTO_IPV6:
		ab.PutBytesAligned32(ip.To16(), 16)
	default:
		return fmt.Errorf("marshal IP: unsupported table family %d", fam)
	}
	return nil
}

func putIPv46Mask(ab *alignedbuff.AlignedBuff, fam TableFamily, mask net.IPMask) error {
	return putIPv46(ab, fam, net.IP(mask))
}
