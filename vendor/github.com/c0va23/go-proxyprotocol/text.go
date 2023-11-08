package proxyprotocol

// TextSignature is prefix for proxyprotocol v1
var (
	TextSignature = []byte("PROXY")
	TextSeparator = " "
	TextCR        = byte('\r')
	TextLF        = byte('\n')
	TextCRLF      = []byte{TextCR, TextLF}
)

var (
	textSignatureLen    = len(TextSignature)
	textAddressPartsLen = 4
	textPortBitSize     = 16
)

// TextProtocol list
var (
	TextProtocolIPv4    = "TCP4"
	TextProtocolIPv6    = "TCP6"
	TextProtocolUnknown = "UNKNOWN"
)
