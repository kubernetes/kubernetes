package proxyprotocol

// BinarySignature is magic prefix for proxyprotocol Binary
var (
	BinarySignature    = []byte{0x0D, 0x0A, 0x0D, 0x0A, 0x00, 0x0D, 0x0A, 0x51, 0x55, 0x49, 0x54, 0x0A}
	BinarySignatureLen = len(BinarySignature)
)

// BinaryVersion2 bits
const (
	BinaryVersion2    byte = 0x20
	BinaryVersionMask byte = 0xF0
)

// Commands
const (
	BinaryCommandLocal byte = 0x00
	BinaryCommandProxy byte = 0x01
	BinaryCommandMask  byte = 0x0F
)

// Address families
const (
	BinaryAFUnspec byte = 0x00
	BinaryAFInet   byte = 0x10
	BinaryAFInet6  byte = 0x20
	BinaryAFUnix   byte = 0x30
	BinaryAFMask   byte = 0xF0
)

// Transport protocols
const (
	BinaryTPUnspec byte = 0x00
	BinaryTPStream byte = 0x01
	BinaryTPDgram  byte = 0x02
	BinaryTPMask   byte = 0x0F
)

// Protocol variants
var (
	BinaryProtocolUnspec       = BinaryAFUnspec | BinaryTPUnspec
	BinaryProtocolTCPoverIPv4  = BinaryAFInet | BinaryTPStream
	BinaryProtocolUDPoverIPv4  = BinaryAFInet | BinaryTPDgram
	BinaryProtocolTCPoverIPv6  = BinaryAFInet6 | BinaryTPStream
	BinaryProtocolUDPoverIPv6  = BinaryAFInet6 | BinaryTPDgram
	BinaryProtocolUnixStream   = BinaryAFUnix | BinaryTPStream
	BinaryProtocolUnixDatagram = BinaryAFUnix | BinaryTPDgram
)

// Expected address length
var (
	BinaryPortLen = 2
)

// TLV types
const (
	TLVTypeNoop byte = 0x04
)
