package proxyproto

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"net"
)

var (
	lengthUnspec      = uint16(0)
	lengthV4          = uint16(12)
	lengthV6          = uint16(36)
	lengthUnix        = uint16(216)
	lengthUnspecBytes = func() []byte {
		a := make([]byte, 2)
		binary.BigEndian.PutUint16(a, lengthUnspec)
		return a
	}()
	lengthV4Bytes = func() []byte {
		a := make([]byte, 2)
		binary.BigEndian.PutUint16(a, lengthV4)
		return a
	}()
	lengthV6Bytes = func() []byte {
		a := make([]byte, 2)
		binary.BigEndian.PutUint16(a, lengthV6)
		return a
	}()
	lengthUnixBytes = func() []byte {
		a := make([]byte, 2)
		binary.BigEndian.PutUint16(a, lengthUnix)
		return a
	}()
	errUint16Overflow = errors.New("proxyproto: uint16 overflow")
)

type _ports struct {
	SrcPort uint16
	DstPort uint16
}

type _addr4 struct {
	Src     [4]byte
	Dst     [4]byte
	SrcPort uint16
	DstPort uint16
}

type _addr6 struct {
	Src [16]byte
	Dst [16]byte
	_ports
}

type _addrUnix struct {
	Src [108]byte
	Dst [108]byte
}

func parseVersion2(reader *bufio.Reader) (header *Header, err error) {
	// Skip first 12 bytes (signature)
	for i := 0; i < 12; i++ {
		if _, err = reader.ReadByte(); err != nil {
			return nil, ErrCantReadProtocolVersionAndCommand
		}
	}

	header = new(Header)
	header.Version = 2

	// Read the 13th byte, protocol version and command
	b13, err := reader.ReadByte()
	if err != nil {
		return nil, ErrCantReadProtocolVersionAndCommand
	}
	header.Command = ProtocolVersionAndCommand(b13)
	if _, ok := supportedCommand[header.Command]; !ok {
		return nil, ErrUnsupportedProtocolVersionAndCommand
	}

	// Read the 14th byte, address family and protocol
	b14, err := reader.ReadByte()
	if err != nil {
		return nil, ErrCantReadAddressFamilyAndProtocol
	}
	header.TransportProtocol = AddressFamilyAndProtocol(b14)
	// UNSPEC is only supported when LOCAL is set.
	if header.TransportProtocol == UNSPEC && header.Command != LOCAL {
		return nil, ErrUnsupportedAddressFamilyAndProtocol
	}

	// Make sure there are bytes available as specified in length
	var length uint16
	if err := binary.Read(io.LimitReader(reader, 2), binary.BigEndian, &length); err != nil {
		return nil, ErrCantReadLength
	}
	if !header.validateLength(length) {
		return nil, ErrInvalidLength
	}

	// Return early if the length is zero, which means that
	// there's no address information and TLVs present for UNSPEC.
	if length == 0 {
		return header, nil
	}

	if _, err := reader.Peek(int(length)); err != nil {
		return nil, ErrInvalidLength
	}

	// Length-limited reader for payload section
	payloadReader := io.LimitReader(reader, int64(length)).(*io.LimitedReader)

	// Read addresses and ports for protocols other than UNSPEC.
	// Ignore address information for UNSPEC, and skip straight to read TLVs,
	// since the length is greater than zero.
	if header.TransportProtocol != UNSPEC {
		if header.TransportProtocol.IsIPv4() {
			var addr _addr4
			if err := binary.Read(payloadReader, binary.BigEndian, &addr); err != nil {
				return nil, ErrInvalidAddress
			}
			header.SourceAddr = newIPAddr(header.TransportProtocol, addr.Src[:], addr.SrcPort)
			header.DestinationAddr = newIPAddr(header.TransportProtocol, addr.Dst[:], addr.DstPort)
		} else if header.TransportProtocol.IsIPv6() {
			var addr _addr6
			if err := binary.Read(payloadReader, binary.BigEndian, &addr); err != nil {
				return nil, ErrInvalidAddress
			}
			header.SourceAddr = newIPAddr(header.TransportProtocol, addr.Src[:], addr.SrcPort)
			header.DestinationAddr = newIPAddr(header.TransportProtocol, addr.Dst[:], addr.DstPort)
		} else if header.TransportProtocol.IsUnix() {
			var addr _addrUnix
			if err := binary.Read(payloadReader, binary.BigEndian, &addr); err != nil {
				return nil, ErrInvalidAddress
			}

			network := "unix"
			if header.TransportProtocol.IsDatagram() {
				network = "unixgram"
			}

			header.SourceAddr = &net.UnixAddr{
				Net:  network,
				Name: parseUnixName(addr.Src[:]),
			}
			header.DestinationAddr = &net.UnixAddr{
				Net:  network,
				Name: parseUnixName(addr.Dst[:]),
			}
		}
	}

	// Copy bytes for optional Type-Length-Value vector
	header.rawTLVs = make([]byte, payloadReader.N) // Allocate minimum size slice
	if _, err = io.ReadFull(payloadReader, header.rawTLVs); err != nil && err != io.EOF {
		return nil, err
	}

	return header, nil
}

func (header *Header) formatVersion2() ([]byte, error) {
	var buf bytes.Buffer
	buf.Write(SIGV2)
	buf.WriteByte(header.Command.toByte())
	buf.WriteByte(header.TransportProtocol.toByte())
	if header.TransportProtocol.IsUnspec() {
		// For UNSPEC, write no addresses and ports but only TLVs if they are present
		hdrLen, err := addTLVLen(lengthUnspecBytes, len(header.rawTLVs))
		if err != nil {
			return nil, err
		}
		buf.Write(hdrLen)
	} else {
		var addrSrc, addrDst []byte
		if header.TransportProtocol.IsIPv4() {
			hdrLen, err := addTLVLen(lengthV4Bytes, len(header.rawTLVs))
			if err != nil {
				return nil, err
			}
			buf.Write(hdrLen)
			sourceIP, destIP, _ := header.IPs()
			addrSrc = sourceIP.To4()
			addrDst = destIP.To4()
		} else if header.TransportProtocol.IsIPv6() {
			hdrLen, err := addTLVLen(lengthV6Bytes, len(header.rawTLVs))
			if err != nil {
				return nil, err
			}
			buf.Write(hdrLen)
			sourceIP, destIP, _ := header.IPs()
			addrSrc = sourceIP.To16()
			addrDst = destIP.To16()
		} else if header.TransportProtocol.IsUnix() {
			buf.Write(lengthUnixBytes)
			sourceAddr, destAddr, ok := header.UnixAddrs()
			if !ok {
				return nil, ErrInvalidAddress
			}
			addrSrc = formatUnixName(sourceAddr.Name)
			addrDst = formatUnixName(destAddr.Name)
		}

		if addrSrc == nil || addrDst == nil {
			return nil, ErrInvalidAddress
		}
		buf.Write(addrSrc)
		buf.Write(addrDst)

		if sourcePort, destPort, ok := header.Ports(); ok {
			portBytes := make([]byte, 2)

			binary.BigEndian.PutUint16(portBytes, uint16(sourcePort))
			buf.Write(portBytes)

			binary.BigEndian.PutUint16(portBytes, uint16(destPort))
			buf.Write(portBytes)
		}
	}

	if len(header.rawTLVs) > 0 {
		buf.Write(header.rawTLVs)
	}

	return buf.Bytes(), nil
}

func (header *Header) validateLength(length uint16) bool {
	if header.TransportProtocol.IsIPv4() {
		return length >= lengthV4
	} else if header.TransportProtocol.IsIPv6() {
		return length >= lengthV6
	} else if header.TransportProtocol.IsUnix() {
		return length >= lengthUnix
	} else if header.TransportProtocol.IsUnspec() {
		return length >= lengthUnspec
	}
	return false
}

// addTLVLen adds the length of the TLV to the header length or errors on uint16 overflow.
func addTLVLen(cur []byte, tlvLen int) ([]byte, error) {
	if tlvLen == 0 {
		return cur, nil
	}
	curLen := binary.BigEndian.Uint16(cur)
	newLen := int(curLen) + tlvLen
	if newLen >= 1<<16 {
		return nil, errUint16Overflow
	}
	a := make([]byte, 2)
	binary.BigEndian.PutUint16(a, uint16(newLen))
	return a, nil
}

func newIPAddr(transport AddressFamilyAndProtocol, ip net.IP, port uint16) net.Addr {
	if transport.IsStream() {
		return &net.TCPAddr{IP: ip, Port: int(port)}
	} else if transport.IsDatagram() {
		return &net.UDPAddr{IP: ip, Port: int(port)}
	} else {
		return nil
	}
}

func parseUnixName(b []byte) string {
	i := bytes.IndexByte(b, 0)
	if i < 0 {
		return string(b)
	}
	return string(b[:i])
}

func formatUnixName(name string) []byte {
	n := int(lengthUnix) / 2
	if len(name) >= n {
		return []byte(name[:n])
	}
	pad := make([]byte, n-len(name))
	return append([]byte(name), pad...)
}
