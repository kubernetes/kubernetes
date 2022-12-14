package ntlmssp

import (
	"bytes"
)

var signature = [8]byte{'N', 'T', 'L', 'M', 'S', 'S', 'P', 0}

type messageHeader struct {
	Signature   [8]byte
	MessageType uint32
}

func (h messageHeader) IsValid() bool {
	return bytes.Equal(h.Signature[:], signature[:]) &&
		h.MessageType > 0 && h.MessageType < 4
}

func newMessageHeader(messageType uint32) messageHeader {
	return messageHeader{signature, messageType}
}
