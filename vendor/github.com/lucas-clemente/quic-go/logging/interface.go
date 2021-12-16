// Package logging defines a logging interface for quic-go.
// This package should not be considered stable
package logging

import (
	"context"
	"net"
	"time"

	"github.com/lucas-clemente/quic-go/internal/utils"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qerr"
	"github.com/lucas-clemente/quic-go/internal/wire"
)

type (
	// A ByteCount is used to count bytes.
	ByteCount = protocol.ByteCount
	// A ConnectionID is a QUIC Connection ID.
	ConnectionID = protocol.ConnectionID
	// The EncryptionLevel is the encryption level of a packet.
	EncryptionLevel = protocol.EncryptionLevel
	// The KeyPhase is the key phase of the 1-RTT keys.
	KeyPhase = protocol.KeyPhase
	// The KeyPhaseBit is the value of the key phase bit of the 1-RTT packets.
	KeyPhaseBit = protocol.KeyPhaseBit
	// The PacketNumber is the packet number of a packet.
	PacketNumber = protocol.PacketNumber
	// The Perspective is the role of a QUIC endpoint (client or server).
	Perspective = protocol.Perspective
	// A StatelessResetToken is a stateless reset token.
	StatelessResetToken = protocol.StatelessResetToken
	// The StreamID is the stream ID.
	StreamID = protocol.StreamID
	// The StreamNum is the number of the stream.
	StreamNum = protocol.StreamNum
	// The StreamType is the type of the stream (unidirectional or bidirectional).
	StreamType = protocol.StreamType
	// The VersionNumber is the QUIC version.
	VersionNumber = protocol.VersionNumber

	// The Header is the QUIC packet header, before removing header protection.
	Header = wire.Header
	// The ExtendedHeader is the QUIC packet header, after removing header protection.
	ExtendedHeader = wire.ExtendedHeader
	// The TransportParameters are QUIC transport parameters.
	TransportParameters = wire.TransportParameters
	// The PreferredAddress is the preferred address sent in the transport parameters.
	PreferredAddress = wire.PreferredAddress

	// A TransportError is a transport-level error code.
	TransportError = qerr.TransportErrorCode
	// An ApplicationError is an application-defined error code.
	ApplicationError = qerr.TransportErrorCode

	// The RTTStats contain statistics used by the congestion controller.
	RTTStats = utils.RTTStats
)

const (
	// KeyPhaseZero is key phase bit 0
	KeyPhaseZero KeyPhaseBit = protocol.KeyPhaseZero
	// KeyPhaseOne is key phase bit 1
	KeyPhaseOne KeyPhaseBit = protocol.KeyPhaseOne
)

const (
	// PerspectiveServer is used for a QUIC server
	PerspectiveServer Perspective = protocol.PerspectiveServer
	// PerspectiveClient is used for a QUIC client
	PerspectiveClient Perspective = protocol.PerspectiveClient
)

const (
	// EncryptionInitial is the Initial encryption level
	EncryptionInitial EncryptionLevel = protocol.EncryptionInitial
	// EncryptionHandshake is the Handshake encryption level
	EncryptionHandshake EncryptionLevel = protocol.EncryptionHandshake
	// Encryption1RTT is the 1-RTT encryption level
	Encryption1RTT EncryptionLevel = protocol.Encryption1RTT
	// Encryption0RTT is the 0-RTT encryption level
	Encryption0RTT EncryptionLevel = protocol.Encryption0RTT
)

const (
	// StreamTypeUni is a unidirectional stream
	StreamTypeUni = protocol.StreamTypeUni
	// StreamTypeBidi is a bidirectional stream
	StreamTypeBidi = protocol.StreamTypeBidi
)

// A Tracer traces events.
type Tracer interface {
	// TracerForConnection requests a new tracer for a connection.
	// The ODCID is the original destination connection ID:
	// The destination connection ID that the client used on the first Initial packet it sent on this connection.
	// If nil is returned, tracing will be disabled for this connection.
	TracerForConnection(ctx context.Context, p Perspective, odcid ConnectionID) ConnectionTracer

	SentPacket(net.Addr, *Header, ByteCount, []Frame)
	DroppedPacket(net.Addr, PacketType, ByteCount, PacketDropReason)
}

// A ConnectionTracer records events.
type ConnectionTracer interface {
	StartedConnection(local, remote net.Addr, srcConnID, destConnID ConnectionID)
	NegotiatedVersion(chosen VersionNumber, clientVersions, serverVersions []VersionNumber)
	ClosedConnection(error)
	SentTransportParameters(*TransportParameters)
	ReceivedTransportParameters(*TransportParameters)
	RestoredTransportParameters(parameters *TransportParameters) // for 0-RTT
	SentPacket(hdr *ExtendedHeader, size ByteCount, ack *AckFrame, frames []Frame)
	ReceivedVersionNegotiationPacket(*Header, []VersionNumber)
	ReceivedRetry(*Header)
	ReceivedPacket(hdr *ExtendedHeader, size ByteCount, frames []Frame)
	BufferedPacket(PacketType)
	DroppedPacket(PacketType, ByteCount, PacketDropReason)
	UpdatedMetrics(rttStats *RTTStats, cwnd, bytesInFlight ByteCount, packetsInFlight int)
	AcknowledgedPacket(EncryptionLevel, PacketNumber)
	LostPacket(EncryptionLevel, PacketNumber, PacketLossReason)
	UpdatedCongestionState(CongestionState)
	UpdatedPTOCount(value uint32)
	UpdatedKeyFromTLS(EncryptionLevel, Perspective)
	UpdatedKey(generation KeyPhase, remote bool)
	DroppedEncryptionLevel(EncryptionLevel)
	DroppedKey(generation KeyPhase)
	SetLossTimer(TimerType, EncryptionLevel, time.Time)
	LossTimerExpired(TimerType, EncryptionLevel)
	LossTimerCanceled()
	// Close is called when the connection is closed.
	Close()
	Debug(name, msg string)
}
