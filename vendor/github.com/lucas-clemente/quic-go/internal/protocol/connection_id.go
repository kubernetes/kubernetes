package protocol

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"io"
)

// A ConnectionID in QUIC
type ConnectionID []byte

const maxConnectionIDLen = 20

// GenerateConnectionID generates a connection ID using cryptographic random
func GenerateConnectionID(len int) (ConnectionID, error) {
	b := make([]byte, len)
	if _, err := rand.Read(b); err != nil {
		return nil, err
	}
	return ConnectionID(b), nil
}

// GenerateConnectionIDForInitial generates a connection ID for the Initial packet.
// It uses a length randomly chosen between 8 and 20 bytes.
func GenerateConnectionIDForInitial() (ConnectionID, error) {
	r := make([]byte, 1)
	if _, err := rand.Read(r); err != nil {
		return nil, err
	}
	len := MinConnectionIDLenInitial + int(r[0])%(maxConnectionIDLen-MinConnectionIDLenInitial+1)
	return GenerateConnectionID(len)
}

// ReadConnectionID reads a connection ID of length len from the given io.Reader.
// It returns io.EOF if there are not enough bytes to read.
func ReadConnectionID(r io.Reader, len int) (ConnectionID, error) {
	if len == 0 {
		return nil, nil
	}
	c := make(ConnectionID, len)
	_, err := io.ReadFull(r, c)
	if err == io.ErrUnexpectedEOF {
		return nil, io.EOF
	}
	return c, err
}

// Equal says if two connection IDs are equal
func (c ConnectionID) Equal(other ConnectionID) bool {
	return bytes.Equal(c, other)
}

// Len returns the length of the connection ID in bytes
func (c ConnectionID) Len() int {
	return len(c)
}

// Bytes returns the byte representation
func (c ConnectionID) Bytes() []byte {
	return []byte(c)
}

func (c ConnectionID) String() string {
	if c.Len() == 0 {
		return "(empty)"
	}
	return fmt.Sprintf("%x", c.Bytes())
}
