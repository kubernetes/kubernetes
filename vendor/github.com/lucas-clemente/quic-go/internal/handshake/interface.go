package handshake

import (
	"errors"
	"io"
	"net"
	"time"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qtls"
	"github.com/lucas-clemente/quic-go/internal/wire"
)

var (
	// ErrKeysNotYetAvailable is returned when an opener or a sealer is requested for an encryption level,
	// but the corresponding opener has not yet been initialized
	// This can happen when packets arrive out of order.
	ErrKeysNotYetAvailable = errors.New("CryptoSetup: keys at this encryption level not yet available")
	// ErrKeysDropped is returned when an opener or a sealer is requested for an encryption level,
	// but the corresponding keys have already been dropped.
	ErrKeysDropped = errors.New("CryptoSetup: keys were already dropped")
	// ErrDecryptionFailed is returned when the AEAD fails to open the packet.
	ErrDecryptionFailed = errors.New("decryption failed")
)

// ConnectionState contains information about the state of the connection.
type ConnectionState = qtls.ConnectionState

type headerDecryptor interface {
	DecryptHeader(sample []byte, firstByte *byte, pnBytes []byte)
}

// LongHeaderOpener opens a long header packet
type LongHeaderOpener interface {
	headerDecryptor
	DecodePacketNumber(wirePN protocol.PacketNumber, wirePNLen protocol.PacketNumberLen) protocol.PacketNumber
	Open(dst, src []byte, pn protocol.PacketNumber, associatedData []byte) ([]byte, error)
}

// ShortHeaderOpener opens a short header packet
type ShortHeaderOpener interface {
	headerDecryptor
	DecodePacketNumber(wirePN protocol.PacketNumber, wirePNLen protocol.PacketNumberLen) protocol.PacketNumber
	Open(dst, src []byte, rcvTime time.Time, pn protocol.PacketNumber, kp protocol.KeyPhaseBit, associatedData []byte) ([]byte, error)
}

// LongHeaderSealer seals a long header packet
type LongHeaderSealer interface {
	Seal(dst, src []byte, packetNumber protocol.PacketNumber, associatedData []byte) []byte
	EncryptHeader(sample []byte, firstByte *byte, pnBytes []byte)
	Overhead() int
}

// ShortHeaderSealer seals a short header packet
type ShortHeaderSealer interface {
	LongHeaderSealer
	KeyPhase() protocol.KeyPhaseBit
}

// A tlsExtensionHandler sends and received the QUIC TLS extension.
type tlsExtensionHandler interface {
	GetExtensions(msgType uint8) []qtls.Extension
	ReceivedExtensions(msgType uint8, exts []qtls.Extension)
	TransportParameters() <-chan []byte
}

type handshakeRunner interface {
	OnReceivedParams(*wire.TransportParameters)
	OnHandshakeComplete()
	OnError(error)
	DropKeys(protocol.EncryptionLevel)
}

// CryptoSetup handles the handshake and protecting / unprotecting packets
type CryptoSetup interface {
	RunHandshake()
	io.Closer
	ChangeConnectionID(protocol.ConnectionID)
	GetSessionTicket() ([]byte, error)

	HandleMessage([]byte, protocol.EncryptionLevel) bool
	SetLargest1RTTAcked(protocol.PacketNumber) error
	SetHandshakeConfirmed()
	ConnectionState() ConnectionState

	GetInitialOpener() (LongHeaderOpener, error)
	GetHandshakeOpener() (LongHeaderOpener, error)
	Get0RTTOpener() (LongHeaderOpener, error)
	Get1RTTOpener() (ShortHeaderOpener, error)

	GetInitialSealer() (LongHeaderSealer, error)
	GetHandshakeSealer() (LongHeaderSealer, error)
	Get0RTTSealer() (LongHeaderSealer, error)
	Get1RTTSealer() (ShortHeaderSealer, error)
}

// ConnWithVersion is the connection used in the ClientHelloInfo.
// It can be used to determine the QUIC version in use.
type ConnWithVersion interface {
	net.Conn
	GetQUICVersion() protocol.VersionNumber
}
