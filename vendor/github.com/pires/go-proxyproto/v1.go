package proxyproto

import (
	"bufio"
	"bytes"
	"fmt"
	"net"
	"strconv"
	"strings"
)

const (
	crlf      = "\r\n"
	separator = " "
)

func initVersion1() *Header {
	header := new(Header)
	header.Version = 1
	// Command doesn't exist in v1
	header.Command = PROXY
	return header
}

func parseVersion1(reader *bufio.Reader) (*Header, error) {
	//The header cannot be more than 107 bytes long. Per spec:
	//
	//   (...)
	//   - worst case (optional fields set to 0xff) :
	//     "PROXY UNKNOWN ffff:f...f:ffff ffff:f...f:ffff 65535 65535\r\n"
	//     => 5 + 1 + 7 + 1 + 39 + 1 + 39 + 1 + 5 + 1 + 5 + 2 = 107 chars
	//
	//   So a 108-byte buffer is always enough to store all the line and a
	//   trailing zero for string processing.
	//
	// It must also be CRLF terminated, as above. The header does not otherwise
	// contain a CR or LF byte.
	//
	// ISSUE #69
	// We can't use Peek here as it will block trying to fill the buffer, which
	// will never happen if the header is TCP4 or TCP6 (max. 56 and 104 bytes
	// respectively) and the server is expected to speak first.
	//
	// Similarly, we can't use ReadString or ReadBytes as these will keep reading
	// until the delimiter is found; an abusive client could easily disrupt a
	// server by sending a large amount of data that do not contain a LF byte.
	// Another means of attack would be to start connections and simply not send
	// data after the initial PROXY signature bytes, accumulating a large
	// number of blocked goroutines on the server. ReadSlice will also block for
	// a delimiter when the internal buffer does not fill up.
	//
	// A plain Read is also problematic since we risk reading past the end of the
	// header without being able to easily put the excess bytes back into the reader's
	// buffer (with the current implementation's design).
	//
	// So we use a ReadByte loop, which solves the overflow problem and avoids
	// reading beyond the end of the header. However, we need one more trick to harden
	// against partial header attacks (slow loris) - per spec:
	//
	//    (..) The sender must always ensure that the header is sent at once, so that
	//    the transport layer maintains atomicity along the path to the receiver. The
	//    receiver may be tolerant to partial headers or may simply drop the connection
	//    when receiving a partial header. Recommendation is to be tolerant, but
	//    implementation constraints may not always easily permit this.
	//
	// We are subject to such implementation constraints. So we return an error if
	// the header cannot be fully extracted with a single read of the underlying
	// reader.
	buf := make([]byte, 0, 107)
	for {
		b, err := reader.ReadByte()
		if err != nil {
			return nil, fmt.Errorf(ErrCantReadVersion1Header.Error()+": %v", err)
		}
		buf = append(buf, b)
		if b == '\n' {
			// End of header found
			break
		}
		if len(buf) == 107 {
			// No delimiter in first 107 bytes
			return nil, ErrVersion1HeaderTooLong
		}
		if reader.Buffered() == 0 {
			// Header was not buffered in a single read. Since we can't
			// differentiate between genuine slow writers and DoS agents,
			// we abort. On healthy networks, this should never happen.
			return nil, ErrCantReadVersion1Header
		}
	}

	// Check for CR before LF.
	if len(buf) < 2 || buf[len(buf)-2] != '\r' {
		return nil, ErrLineMustEndWithCrlf
	}

	// Check full signature.
	tokens := strings.Split(string(buf[:len(buf)-2]), separator)

	// Expect at least 2 tokens: "PROXY" and the transport protocol.
	if len(tokens) < 2 {
		return nil, ErrCantReadAddressFamilyAndProtocol
	}

	// Read address family and protocol
	var transportProtocol AddressFamilyAndProtocol
	switch tokens[1] {
	case "TCP4":
		transportProtocol = TCPv4
	case "TCP6":
		transportProtocol = TCPv6
	case "UNKNOWN":
		transportProtocol = UNSPEC // doesn't exist in v1 but fits UNKNOWN
	default:
		return nil, ErrCantReadAddressFamilyAndProtocol
	}

	// Expect 6 tokens only when UNKNOWN is not present.
	if transportProtocol != UNSPEC && len(tokens) < 6 {
		return nil, ErrCantReadAddressFamilyAndProtocol
	}

	// When a signature is found, allocate a v1 header with Command set to PROXY.
	// Command doesn't exist in v1 but set it for other parts of this library
	// to rely on it for determining connection details.
	header := initVersion1()

	// Transport protocol has been processed already.
	header.TransportProtocol = transportProtocol

	// When UNKNOWN, set the command to LOCAL and return early
	if header.TransportProtocol == UNSPEC {
		header.Command = LOCAL
		return header, nil
	}

	// Otherwise, continue to read addresses and ports
	sourceIP, err := parseV1IPAddress(header.TransportProtocol, tokens[2])
	if err != nil {
		return nil, err
	}
	destIP, err := parseV1IPAddress(header.TransportProtocol, tokens[3])
	if err != nil {
		return nil, err
	}
	sourcePort, err := parseV1PortNumber(tokens[4])
	if err != nil {
		return nil, err
	}
	destPort, err := parseV1PortNumber(tokens[5])
	if err != nil {
		return nil, err
	}
	header.SourceAddr = &net.TCPAddr{
		IP:   sourceIP,
		Port: sourcePort,
	}
	header.DestinationAddr = &net.TCPAddr{
		IP:   destIP,
		Port: destPort,
	}

	return header, nil
}

func (header *Header) formatVersion1() ([]byte, error) {
	// As of version 1, only "TCP4" ( \x54 \x43 \x50 \x34 ) for TCP over IPv4,
	// and "TCP6" ( \x54 \x43 \x50 \x36 ) for TCP over IPv6 are allowed.
	var proto string
	switch header.TransportProtocol {
	case TCPv4:
		proto = "TCP4"
	case TCPv6:
		proto = "TCP6"
	default:
		// Unknown connection (short form)
		return []byte("PROXY UNKNOWN" + crlf), nil
	}

	sourceAddr, sourceOK := header.SourceAddr.(*net.TCPAddr)
	destAddr, destOK := header.DestinationAddr.(*net.TCPAddr)
	if !sourceOK || !destOK {
		return nil, ErrInvalidAddress
	}

	sourceIP, destIP := sourceAddr.IP, destAddr.IP
	switch header.TransportProtocol {
	case TCPv4:
		sourceIP = sourceIP.To4()
		destIP = destIP.To4()
	case TCPv6:
		sourceIP = sourceIP.To16()
		destIP = destIP.To16()
	}
	if sourceIP == nil || destIP == nil {
		return nil, ErrInvalidAddress
	}

	buf := bytes.NewBuffer(make([]byte, 0, 108))
	buf.Write(SIGV1)
	buf.WriteString(separator)
	buf.WriteString(proto)
	buf.WriteString(separator)
	buf.WriteString(sourceIP.String())
	buf.WriteString(separator)
	buf.WriteString(destIP.String())
	buf.WriteString(separator)
	buf.WriteString(strconv.Itoa(sourceAddr.Port))
	buf.WriteString(separator)
	buf.WriteString(strconv.Itoa(destAddr.Port))
	buf.WriteString(crlf)

	return buf.Bytes(), nil
}

func parseV1PortNumber(portStr string) (int, error) {
	port, err := strconv.Atoi(portStr)
	if err != nil || port < 0 || port > 65535 {
		return 0, ErrInvalidPortNumber
	}
	return port, nil
}

func parseV1IPAddress(protocol AddressFamilyAndProtocol, addrStr string) (addr net.IP, err error) {
	addr = net.ParseIP(addrStr)
	tryV4 := addr.To4()
	if (protocol == TCPv4 && tryV4 == nil) || (protocol == TCPv6 && tryV4 != nil) {
		err = ErrInvalidAddress
	}
	return
}
