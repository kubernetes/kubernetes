package encoding

import (
	"sync"
)

// Decoder decodes the contents of b into a v representation.
// It's primarily used for decoding contents of a file into a map[string]interface{}.
type Decoder interface {
	Decode(b []byte, v interface{}) error
}

const (
	// ErrDecoderNotFound is returned when there is no decoder registered for a format.
	ErrDecoderNotFound = encodingError("decoder not found for this format")

	// ErrDecoderFormatAlreadyRegistered is returned when an decoder is already registered for a format.
	ErrDecoderFormatAlreadyRegistered = encodingError("decoder already registered for this format")
)

// DecoderRegistry can choose an appropriate Decoder based on the provided format.
type DecoderRegistry struct {
	decoders map[string]Decoder

	mu sync.RWMutex
}

// NewDecoderRegistry returns a new, initialized DecoderRegistry.
func NewDecoderRegistry() *DecoderRegistry {
	return &DecoderRegistry{
		decoders: make(map[string]Decoder),
	}
}

// RegisterDecoder registers a Decoder for a format.
// Registering a Decoder for an already existing format is not supported.
func (e *DecoderRegistry) RegisterDecoder(format string, enc Decoder) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, ok := e.decoders[format]; ok {
		return ErrDecoderFormatAlreadyRegistered
	}

	e.decoders[format] = enc

	return nil
}

// Decode calls the underlying Decoder based on the format.
func (e *DecoderRegistry) Decode(format string, b []byte, v interface{}) error {
	e.mu.RLock()
	decoder, ok := e.decoders[format]
	e.mu.RUnlock()

	if !ok {
		return ErrDecoderNotFound
	}

	return decoder.Decode(b, v)
}
