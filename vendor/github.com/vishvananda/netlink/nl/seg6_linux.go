package nl

import (
	"errors"
	"fmt"
	"net"
)

type IPv6SrHdr struct {
	nextHdr      uint8
	hdrLen       uint8
	routingType  uint8
	segmentsLeft uint8
	firstSegment uint8
	flags        uint8
	reserved     uint16

	Segments []net.IP
}

func (s1 *IPv6SrHdr) Equal(s2 IPv6SrHdr) bool {
	if len(s1.Segments) != len(s2.Segments) {
		return false
	}
	for i := range s1.Segments {
		if s1.Segments[i].Equal(s2.Segments[i]) != true {
			return false
		}
	}
	return s1.nextHdr == s2.nextHdr &&
		s1.hdrLen == s2.hdrLen &&
		s1.routingType == s2.routingType &&
		s1.segmentsLeft == s2.segmentsLeft &&
		s1.firstSegment == s2.firstSegment &&
		s1.flags == s2.flags
	// reserved doesn't need to be identical.
}

// seg6 encap mode
const (
	SEG6_IPTUN_MODE_INLINE = iota
	SEG6_IPTUN_MODE_ENCAP
)

// number of nested RTATTR
// from include/uapi/linux/seg6_iptunnel.h
const (
	SEG6_IPTUNNEL_UNSPEC = iota
	SEG6_IPTUNNEL_SRH
	__SEG6_IPTUNNEL_MAX
)
const (
	SEG6_IPTUNNEL_MAX = __SEG6_IPTUNNEL_MAX - 1
)

func EncodeSEG6Encap(mode int, segments []net.IP) ([]byte, error) {
	nsegs := len(segments) // nsegs: number of segments
	if nsegs == 0 {
		return nil, errors.New("EncodeSEG6Encap: No Segment in srh")
	}
	b := make([]byte, 12, 12+len(segments)*16)
	native := NativeEndian()
	native.PutUint32(b, uint32(mode))
	b[4] = 0                      // srh.nextHdr (0 when calling netlink)
	b[5] = uint8(16 * nsegs >> 3) // srh.hdrLen (in 8-octets unit)
	b[6] = IPV6_SRCRT_TYPE_4      // srh.routingType (assigned by IANA)
	b[7] = uint8(nsegs - 1)       // srh.segmentsLeft
	b[8] = uint8(nsegs - 1)       // srh.firstSegment
	b[9] = 0                      // srh.flags (SR6_FLAG1_HMAC for srh_hmac)
	// srh.reserved: Defined as "Tag" in draft-ietf-6man-segment-routing-header-07
	native.PutUint16(b[10:], 0) // srh.reserved
	for _, netIP := range segments {
		b = append(b, netIP...) // srh.Segments
	}
	return b, nil
}

func DecodeSEG6Encap(buf []byte) (int, []net.IP, error) {
	native := NativeEndian()
	mode := int(native.Uint32(buf))
	srh := IPv6SrHdr{
		nextHdr:      buf[4],
		hdrLen:       buf[5],
		routingType:  buf[6],
		segmentsLeft: buf[7],
		firstSegment: buf[8],
		flags:        buf[9],
		reserved:     native.Uint16(buf[10:12]),
	}
	buf = buf[12:]
	if len(buf)%16 != 0 {
		err := fmt.Errorf("DecodeSEG6Encap: error parsing Segment List (buf len: %d)\n", len(buf))
		return mode, nil, err
	}
	for len(buf) > 0 {
		srh.Segments = append(srh.Segments, net.IP(buf[:16]))
		buf = buf[16:]
	}
	return mode, srh.Segments, nil
}

func DecodeSEG6Srh(buf []byte) ([]net.IP, error) {
	native := NativeEndian()
	srh := IPv6SrHdr{
		nextHdr:      buf[0],
		hdrLen:       buf[1],
		routingType:  buf[2],
		segmentsLeft: buf[3],
		firstSegment: buf[4],
		flags:        buf[5],
		reserved:     native.Uint16(buf[6:8]),
	}
	buf = buf[8:]
	if len(buf)%16 != 0 {
		err := fmt.Errorf("DecodeSEG6Srh: error parsing Segment List (buf len: %d)", len(buf))
		return nil, err
	}
	for len(buf) > 0 {
		srh.Segments = append(srh.Segments, net.IP(buf[:16]))
		buf = buf[16:]
	}
	return srh.Segments, nil
}
func EncodeSEG6Srh(segments []net.IP) ([]byte, error) {
	nsegs := len(segments) // nsegs: number of segments
	if nsegs == 0 {
		return nil, errors.New("EncodeSEG6Srh: No Segments")
	}
	b := make([]byte, 8, 8+len(segments)*16)
	native := NativeEndian()
	b[0] = 0                      // srh.nextHdr (0 when calling netlink)
	b[1] = uint8(16 * nsegs >> 3) // srh.hdrLen (in 8-octets unit)
	b[2] = IPV6_SRCRT_TYPE_4      // srh.routingType (assigned by IANA)
	b[3] = uint8(nsegs - 1)       // srh.segmentsLeft
	b[4] = uint8(nsegs - 1)       // srh.firstSegment
	b[5] = 0                      // srh.flags (SR6_FLAG1_HMAC for srh_hmac)
	// srh.reserved: Defined as "Tag" in draft-ietf-6man-segment-routing-header-07
	native.PutUint16(b[6:], 0) // srh.reserved
	for _, netIP := range segments {
		b = append(b, netIP...) // srh.Segments
	}
	return b, nil
}

// Helper functions
func SEG6EncapModeString(mode int) string {
	switch mode {
	case SEG6_IPTUN_MODE_INLINE:
		return "inline"
	case SEG6_IPTUN_MODE_ENCAP:
		return "encap"
	}
	return "unknown"
}
