package proxyprotocol

import (
	"bufio"
	"bytes"
	"errors"
	"net"
	"strconv"
	"strings"
)

// Text protocol errors
var (
	ErrInvalidAddressList = errors.New("invalid address list")
	ErrInvalidIP          = errors.New("invalid IP")
	ErrInvalidPort        = errors.New("invalid port")
)

// TextHeaderParser for proxyprotocol v1
type TextHeaderParser struct {
	logger Logger
}

// NewTextHeaderParser construct TextHeaderParser
func NewTextHeaderParser(logger Logger) TextHeaderParser {
	return TextHeaderParser{
		logger: logger,
	}
}

// Parse proxyprotocol v1 header
func (parser TextHeaderParser) Parse(buf *bufio.Reader) (*Header, error) {
	signatureBuf, err := buf.Peek(textSignatureLen)
	if err != nil {
		parser.logger.Printf("Read text signature error: %s", err)
		return nil, err
	}

	if !bytes.Equal(signatureBuf, TextSignature) {
		return nil, ErrInvalidSignature
	}

	headerLine, err := buf.ReadString(TextLF)
	if err != nil {
		parser.logger.Printf("Read header line error: %s", err)
		return nil, err
	}

	// Strip CR char on line end
	if headerLine[len(headerLine)-2] == TextCR {
		headerLine = headerLine[:len(headerLine)-2]
	}

	headerParts := strings.Split(headerLine, TextSeparator)

	protocol := headerParts[1]

	switch protocol {
	case TextProtocolUnknown:
		return nil, nil
	case TextProtocolIPv4, TextProtocolIPv6:
		return parseTextHeader(headerParts)
	default:
		return nil, ErrUnknownProtocol
	}
}

func parseTextHeader(headerParts []string) (*Header, error) {
	addressParts := headerParts[2:]
	if textAddressPartsLen != len(addressParts) {
		return nil, ErrInvalidAddressList
	}

	srcIPStr := addressParts[0]
	srcIP := net.ParseIP(srcIPStr)
	if srcIP == nil {
		return nil, ErrInvalidIP
	}

	dstIPStr := addressParts[1]
	dstIP := net.ParseIP(dstIPStr)
	if dstIP == nil {
		return nil, ErrInvalidIP
	}

	srcPortSrt := addressParts[2]
	srcPort, err := strconv.ParseUint(srcPortSrt, 10, textPortBitSize)
	if err != nil {
		return nil, ErrInvalidPort
	}

	dstPortSrt := addressParts[3]
	dstPort, err := strconv.ParseUint(dstPortSrt, 10, textPortBitSize)
	if err != nil {
		return nil, ErrInvalidPort
	}

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
