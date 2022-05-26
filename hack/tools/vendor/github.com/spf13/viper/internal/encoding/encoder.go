package encoding

import (
	"sync"
)

// Encoder encodes the contents of v into a byte representation.
// It's primarily used for encoding a map[string]interface{} into a file format.
type Encoder interface {
	Encode(v interface{}) ([]byte, error)
}

const (
	// ErrEncoderNotFound is returned when there is no encoder registered for a format.
	ErrEncoderNotFound = encodingError("encoder not found for this format")

	// ErrEncoderFormatAlreadyRegistered is returned when an encoder is already registered for a format.
	ErrEncoderFormatAlreadyRegistered = encodingError("encoder already registered for this format")
)

// EncoderRegistry can choose an appropriate Encoder based on the provided format.
type EncoderRegistry struct {
	encoders map[string]Encoder

	mu sync.RWMutex
}

// NewEncoderRegistry returns a new, initialized EncoderRegistry.
func NewEncoderRegistry() *EncoderRegistry {
	return &EncoderRegistry{
		encoders: make(map[string]Encoder),
	}
}

// RegisterEncoder registers an Encoder for a format.
// Registering a Encoder for an already existing format is not supported.
func (e *EncoderRegistry) RegisterEncoder(format string, enc Encoder) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, ok := e.encoders[format]; ok {
		return ErrEncoderFormatAlreadyRegistered
	}

	e.encoders[format] = enc

	return nil
}

func (e *EncoderRegistry) Encode(format string, v interface{}) ([]byte, error) {
	e.mu.RLock()
	encoder, ok := e.encoders[format]
	e.mu.RUnlock()

	if !ok {
		return nil, ErrEncoderNotFound
	}

	return encoder.Encode(v)
}
