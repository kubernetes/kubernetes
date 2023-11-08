package proxyprotocol

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"net"
)

// Errors
var (
	ErrUnknownVersion       = errors.New("unknown version")
	ErrUnknownCommand       = errors.New("unknown command")
	ErrUnexpectedAddressLen = errors.New("unexpected address length")
)

// Meta buffer byte position
const (
	versionCommandPos  = 0
	protocolPos        = 1
	addressLenStartPos = 2
	addressLenEndPos   = 4
)

// BinaryHeaderParser parse proxyprotocol header from Reader
type BinaryHeaderParser struct {
	logger Logger
}

// NewBinaryHeaderParser construct BinaryHeaderParser
func NewBinaryHeaderParser(logger Logger) BinaryHeaderParser {
	return BinaryHeaderParser{
		logger: logger,
	}
}

// Parse buffer
func (parser BinaryHeaderParser) Parse(buf *bufio.Reader) (*Header, error) {
	magicBuf, err := buf.Peek(BinarySignatureLen)
	if err != nil {
		parser.logger.Printf("Read magic prefix error: %s", err)
		return nil, err
	}

	if !bytes.Equal(magicBuf, BinarySignature) {
		return nil, ErrInvalidSignature
	}

	_, err = buf.Discard(BinarySignatureLen)
	if err != nil {
		return nil, err
	}

	metaBuf := make([]byte, addressLenEndPos)
	if _, err = buf.Read(metaBuf); err != nil {
		parser.logger.Printf("Read meta error: %s", err)
		return nil, err
	}

	versionCommandByte := metaBuf[versionCommandPos]

	if versionCommandByte&BinaryVersionMask != BinaryVersion2 {
		return nil, ErrUnknownVersion
	}

	addressSizeBuf := metaBuf[addressLenStartPos:addressLenEndPos]
	addressesLen := int(binary.BigEndian.Uint16(addressSizeBuf))
	parser.logger.Printf("Addresses len: %d", addressesLen)

	addressesBuf := make([]byte, addressesLen)
	addressReaded, err := buf.Read(addressesBuf)
	if err != nil {
		parser.logger.Printf("Read address error: %s", err)
		return nil, err
	}
	parser.logger.Printf("Address readed: %d", addressReaded)

	switch versionCommandByte & BinaryCommandMask {
	case BinaryCommandProxy:
		return parserBinaryCommandHeader(metaBuf[protocolPos], addressesBuf)
	case BinaryCommandLocal:
		return nil, nil
	default:
		return nil, ErrUnknownCommand
	}
}

func parserBinaryCommandHeader(protocol byte, addressesBuf []byte) (*Header, error) {
	switch protocol & BinaryAFMask {
	case BinaryProtocolUnspec:
		return nil, nil
	case BinaryAFInet:
		return parseAddressData(addressesBuf, net.IPv4len)
	case BinaryAFInet6:
		return parseAddressData(addressesBuf, net.IPv6len)
	default:
		return nil, ErrUnknownProtocol
	}
}

func parseAddressData(addressesBuf []byte, ipLen int) (*Header, error) {
	expectedBufSize := 2 * (ipLen + BinaryPortLen)
	if len(addressesBuf) < expectedBufSize {
		return nil, ErrUnexpectedAddressLen
	}

	srcIP := make(net.IP, ipLen)
	copy(srcIP, addressesBuf[:ipLen])
	addressesBuf = addressesBuf[ipLen:]

	dstIP := make(net.IP, ipLen)
	copy(dstIP, addressesBuf[:ipLen])
	addressesBuf = addressesBuf[ipLen:]

	srcPort := binary.BigEndian.Uint16(addressesBuf[:BinaryPortLen])
	addressesBuf = addressesBuf[BinaryPortLen:]

	dstPort := binary.BigEndian.Uint16(addressesBuf[:BinaryPortLen])
	// addressesBuf = addressesBuf[BinaryPortLen:]

	return &Header{
		SrcAddr: &net.TCPAddr{
			IP:   srcIP,
			Port: int(srcPort),
		},
		DstAddr: &net.TCPAddr{
			IP:   dstIP,
			Port: int(dstPort),
		},
	}, nil
}
