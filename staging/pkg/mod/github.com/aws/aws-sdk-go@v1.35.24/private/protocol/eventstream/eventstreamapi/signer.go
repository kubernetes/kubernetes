package eventstreamapi

import (
	"bytes"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/private/protocol/eventstream"
)

var timeNow = time.Now

// StreamSigner defines an interface for the implementation of signing of event stream payloads
type StreamSigner interface {
	GetSignature(headers, payload []byte, date time.Time) ([]byte, error)
}

// SignEncoder envelopes event stream messages
// into an event stream message payload with included
// signature headers using the provided signer and encoder.
type SignEncoder struct {
	signer     StreamSigner
	encoder    Encoder
	bufEncoder *BufferEncoder

	closeErr error
	closed   bool
}

// NewSignEncoder returns a new SignEncoder using the provided stream signer and
// event stream encoder.
func NewSignEncoder(signer StreamSigner, encoder Encoder) *SignEncoder {
	// TODO: Need to pass down logging

	return &SignEncoder{
		signer:     signer,
		encoder:    encoder,
		bufEncoder: NewBufferEncoder(),
	}
}

// Close encodes a final event stream signing envelope with an empty event stream
// payload. This final end-frame is used to mark the conclusion of the stream.
func (s *SignEncoder) Close() error {
	if s.closed {
		return s.closeErr
	}

	if err := s.encode([]byte{}); err != nil {
		if strings.Contains(err.Error(), "on closed pipe") {
			return nil
		}

		s.closeErr = err
		s.closed = true
		return s.closeErr
	}

	return nil
}

// Encode takes the provided message and add envelopes the message
// with the required signature.
func (s *SignEncoder) Encode(msg eventstream.Message) error {
	payload, err := s.bufEncoder.Encode(msg)
	if err != nil {
		return err
	}

	return s.encode(payload)
}

func (s SignEncoder) encode(payload []byte) error {
	date := timeNow()

	var msg eventstream.Message
	msg.Headers.Set(DateHeader, eventstream.TimestampValue(date))
	msg.Payload = payload

	var headers bytes.Buffer
	if err := eventstream.EncodeHeaders(&headers, msg.Headers); err != nil {
		return err
	}

	sig, err := s.signer.GetSignature(headers.Bytes(), msg.Payload, date)
	if err != nil {
		return err
	}

	msg.Headers.Set(ChunkSignatureHeader, eventstream.BytesValue(sig))

	return s.encoder.Encode(msg)
}

// BufferEncoder is a utility that provides a buffered
// event stream encoder
type BufferEncoder struct {
	encoder Encoder
	buffer  *bytes.Buffer
}

// NewBufferEncoder returns a new BufferEncoder initialized
// with a 1024 byte buffer.
func NewBufferEncoder() *BufferEncoder {
	buf := bytes.NewBuffer(make([]byte, 1024))
	return &BufferEncoder{
		encoder: eventstream.NewEncoder(buf),
		buffer:  buf,
	}
}

// Encode returns the encoded message as a byte slice.
// The returned byte slice will be modified on the next encode call
// and should not be held onto.
func (e *BufferEncoder) Encode(msg eventstream.Message) ([]byte, error) {
	e.buffer.Reset()

	if err := e.encoder.Encode(msg); err != nil {
		return nil, err
	}

	return e.buffer.Bytes(), nil
}
