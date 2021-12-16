package ackhandler

import "fmt"

// The SendMode says what kind of packets can be sent.
type SendMode uint8

const (
	// SendNone means that no packets should be sent
	SendNone SendMode = iota
	// SendAck means an ACK-only packet should be sent
	SendAck
	// SendPTOInitial means that an Initial probe packet should be sent
	SendPTOInitial
	// SendPTOHandshake means that a Handshake probe packet should be sent
	SendPTOHandshake
	// SendPTOAppData means that an Application data probe packet should be sent
	SendPTOAppData
	// SendAny means that any packet should be sent
	SendAny
)

func (s SendMode) String() string {
	switch s {
	case SendNone:
		return "none"
	case SendAck:
		return "ack"
	case SendPTOInitial:
		return "pto (Initial)"
	case SendPTOHandshake:
		return "pto (Handshake)"
	case SendPTOAppData:
		return "pto (Application Data)"
	case SendAny:
		return "any"
	default:
		return fmt.Sprintf("invalid send mode: %d", s)
	}
}
