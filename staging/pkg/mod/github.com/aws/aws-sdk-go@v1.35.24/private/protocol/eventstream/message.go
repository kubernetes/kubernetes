package eventstream

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
)

const preludeLen = 8
const preludeCRCLen = 4
const msgCRCLen = 4
const minMsgLen = preludeLen + preludeCRCLen + msgCRCLen
const maxPayloadLen = 1024 * 1024 * 16 // 16MB
const maxHeadersLen = 1024 * 128       // 128KB
const maxMsgLen = minMsgLen + maxHeadersLen + maxPayloadLen

var crc32IEEETable = crc32.MakeTable(crc32.IEEE)

// A Message provides the eventstream message representation.
type Message struct {
	Headers Headers
	Payload []byte
}

func (m *Message) rawMessage() (rawMessage, error) {
	var raw rawMessage

	if len(m.Headers) > 0 {
		var headers bytes.Buffer
		if err := EncodeHeaders(&headers, m.Headers); err != nil {
			return rawMessage{}, err
		}
		raw.Headers = headers.Bytes()
		raw.HeadersLen = uint32(len(raw.Headers))
	}

	raw.Length = raw.HeadersLen + uint32(len(m.Payload)) + minMsgLen

	hash := crc32.New(crc32IEEETable)
	binaryWriteFields(hash, binary.BigEndian, raw.Length, raw.HeadersLen)
	raw.PreludeCRC = hash.Sum32()

	binaryWriteFields(hash, binary.BigEndian, raw.PreludeCRC)

	if raw.HeadersLen > 0 {
		hash.Write(raw.Headers)
	}

	// Read payload bytes and update hash for it as well.
	if len(m.Payload) > 0 {
		raw.Payload = m.Payload
		hash.Write(raw.Payload)
	}

	raw.CRC = hash.Sum32()

	return raw, nil
}

// Clone returns a deep copy of the message.
func (m Message) Clone() Message {
	var payload []byte
	if m.Payload != nil {
		payload = make([]byte, len(m.Payload))
		copy(payload, m.Payload)
	}

	return Message{
		Headers: m.Headers.Clone(),
		Payload: payload,
	}
}

type messagePrelude struct {
	Length     uint32
	HeadersLen uint32
	PreludeCRC uint32
}

func (p messagePrelude) PayloadLen() uint32 {
	return p.Length - p.HeadersLen - minMsgLen
}

func (p messagePrelude) ValidateLens() error {
	if p.Length == 0 || p.Length > maxMsgLen {
		return LengthError{
			Part: "message prelude",
			Want: maxMsgLen,
			Have: int(p.Length),
		}
	}
	if p.HeadersLen > maxHeadersLen {
		return LengthError{
			Part: "message headers",
			Want: maxHeadersLen,
			Have: int(p.HeadersLen),
		}
	}
	if payloadLen := p.PayloadLen(); payloadLen > maxPayloadLen {
		return LengthError{
			Part: "message payload",
			Want: maxPayloadLen,
			Have: int(payloadLen),
		}
	}

	return nil
}

type rawMessage struct {
	messagePrelude

	Headers []byte
	Payload []byte

	CRC uint32
}
