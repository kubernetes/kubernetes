package ackhandler

import (
	"time"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/wire"
)

// A Packet is a packet
type Packet struct {
	PacketNumber    protocol.PacketNumber
	Frames          []Frame
	LargestAcked    protocol.PacketNumber // InvalidPacketNumber if the packet doesn't contain an ACK
	Length          protocol.ByteCount
	EncryptionLevel protocol.EncryptionLevel
	SendTime        time.Time

	IsPathMTUProbePacket bool // We don't report the loss of Path MTU probe packets to the congestion controller.

	includedInBytesInFlight bool
	declaredLost            bool
	skippedPacket           bool
}

// SentPacketHandler handles ACKs received for outgoing packets
type SentPacketHandler interface {
	// SentPacket may modify the packet
	SentPacket(packet *Packet)
	ReceivedAck(ackFrame *wire.AckFrame, encLevel protocol.EncryptionLevel, recvTime time.Time) (bool /* 1-RTT packet acked */, error)
	ReceivedBytes(protocol.ByteCount)
	DropPackets(protocol.EncryptionLevel)
	ResetForRetry() error
	SetHandshakeConfirmed()

	// The SendMode determines if and what kind of packets can be sent.
	SendMode() SendMode
	// TimeUntilSend is the time when the next packet should be sent.
	// It is used for pacing packets.
	TimeUntilSend() time.Time
	// HasPacingBudget says if the pacer allows sending of a (full size) packet at this moment.
	HasPacingBudget() bool
	SetMaxDatagramSize(count protocol.ByteCount)

	// only to be called once the handshake is complete
	QueueProbePacket(protocol.EncryptionLevel) bool /* was a packet queued */

	PeekPacketNumber(protocol.EncryptionLevel) (protocol.PacketNumber, protocol.PacketNumberLen)
	PopPacketNumber(protocol.EncryptionLevel) protocol.PacketNumber

	GetLossDetectionTimeout() time.Time
	OnLossDetectionTimeout() error
}

type sentPacketTracker interface {
	GetLowestPacketNotConfirmedAcked() protocol.PacketNumber
	ReceivedPacket(protocol.EncryptionLevel)
}

// ReceivedPacketHandler handles ACKs needed to send for incoming packets
type ReceivedPacketHandler interface {
	IsPotentiallyDuplicate(protocol.PacketNumber, protocol.EncryptionLevel) bool
	ReceivedPacket(pn protocol.PacketNumber, ecn protocol.ECN, encLevel protocol.EncryptionLevel, rcvTime time.Time, shouldInstigateAck bool) error
	DropPackets(protocol.EncryptionLevel)

	GetAlarmTimeout() time.Time
	GetAckFrame(encLevel protocol.EncryptionLevel, onlyIfQueued bool) *wire.AckFrame
}
