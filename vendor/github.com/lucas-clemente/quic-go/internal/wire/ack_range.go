package wire

import "github.com/lucas-clemente/quic-go/internal/protocol"

// AckRange is an ACK range
type AckRange struct {
	Smallest protocol.PacketNumber
	Largest  protocol.PacketNumber
}

// Len returns the number of packets contained in this ACK range
func (r AckRange) Len() protocol.PacketNumber {
	return r.Largest - r.Smallest + 1
}
