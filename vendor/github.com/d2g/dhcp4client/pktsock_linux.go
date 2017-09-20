package dhcp4client

import (
	"crypto/rand"
	"encoding/binary"
	"net"
	"time"

	"golang.org/x/sys/unix"
)

const (
	minIPHdrLen = 20
	maxIPHdrLen = 60
	udpHdrLen   = 8
	ip4Ver      = 0x40
	ttl         = 16
	srcPort     = 68
	dstPort     = 67
)

var (
	bcastMAC = []byte{255, 255, 255, 255, 255, 255}
)

// abstracts AF_PACKET
type packetSock struct {
	fd      int
	ifindex int
}

func NewPacketSock(ifindex int) (*packetSock, error) {
	fd, err := unix.Socket(unix.AF_PACKET, unix.SOCK_DGRAM, int(swap16(unix.ETH_P_IP)))
	if err != nil {
		return nil, err
	}

	addr := unix.SockaddrLinklayer{
		Ifindex:  ifindex,
		Protocol: swap16(unix.ETH_P_IP),
	}

	if err = unix.Bind(fd, &addr); err != nil {
		return nil, err
	}

	return &packetSock{
		fd:      fd,
		ifindex: ifindex,
	}, nil
}

func (pc *packetSock) Close() error {
	return unix.Close(pc.fd)
}

func (pc *packetSock) Write(packet []byte) error {
	lladdr := unix.SockaddrLinklayer{
		Ifindex:  pc.ifindex,
		Protocol: swap16(unix.ETH_P_IP),
		Halen:    uint8(len(bcastMAC)),
	}
	copy(lladdr.Addr[:], bcastMAC)

	pkt := make([]byte, minIPHdrLen+udpHdrLen+len(packet))

	fillIPHdr(pkt[0:minIPHdrLen], udpHdrLen+uint16(len(packet)))
	fillUDPHdr(pkt[minIPHdrLen:minIPHdrLen+udpHdrLen], uint16(len(packet)))

	// payload
	copy(pkt[minIPHdrLen+udpHdrLen:len(pkt)], packet)

	return unix.Sendto(pc.fd, pkt, 0, &lladdr)
}

func (pc *packetSock) ReadFrom() ([]byte, net.IP, error) {
	pkt := make([]byte, maxIPHdrLen+udpHdrLen+MaxDHCPLen)
	n, _, err := unix.Recvfrom(pc.fd, pkt, 0)
	if err != nil {
		return nil, nil, err
	}

	// IP hdr len
	ihl := int(pkt[0]&0x0F) * 4
	// Source IP address
	src := net.IP(pkt[12:16])

	return pkt[ihl+udpHdrLen : n], src, nil
}

func (pc *packetSock) SetReadTimeout(t time.Duration) error {

	tv := unix.NsecToTimeval(t.Nanoseconds())
	return unix.SetsockoptTimeval(pc.fd, unix.SOL_SOCKET, unix.SO_RCVTIMEO, &tv)
}

// compute's 1's complement checksum
func chksum(p []byte, csum []byte) {
	cklen := len(p)
	s := uint32(0)
	for i := 0; i < (cklen - 1); i += 2 {
		s += uint32(p[i+1])<<8 | uint32(p[i])
	}
	if cklen&1 == 1 {
		s += uint32(p[cklen-1])
	}
	s = (s >> 16) + (s & 0xffff)
	s = s + (s >> 16)
	s = ^s

	csum[0] = uint8(s & 0xff)
	csum[1] = uint8(s >> 8)
}

func fillIPHdr(hdr []byte, payloadLen uint16) {
	// version + IHL
	hdr[0] = ip4Ver | (minIPHdrLen / 4)
	// total length
	binary.BigEndian.PutUint16(hdr[2:4], uint16(len(hdr))+payloadLen)
	// identification
	if _, err := rand.Read(hdr[4:5]); err != nil {
		panic(err)
	}
	// TTL
	hdr[8] = 16
	// Protocol
	hdr[9] = unix.IPPROTO_UDP
	// dst IP
	copy(hdr[16:20], net.IPv4bcast.To4())
	// compute IP hdr checksum
	chksum(hdr[0:len(hdr)], hdr[10:12])
}

func fillUDPHdr(hdr []byte, payloadLen uint16) {
	// src port
	binary.BigEndian.PutUint16(hdr[0:2], srcPort)
	// dest port
	binary.BigEndian.PutUint16(hdr[2:4], dstPort)
	// length
	binary.BigEndian.PutUint16(hdr[4:6], udpHdrLen+payloadLen)
}

func swap16(x uint16) uint16 {
	var b [2]byte
	binary.BigEndian.PutUint16(b[:], x)
	return binary.LittleEndian.Uint16(b[:])
}
