package types

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"regexp"

	"github.com/akutz/goof"
)

// UUID is a UUID.
//
// This was totally stolen from
// https://github.com/nu7hatch/gouuid/blob/master/uuid.go, and all credit goes
// to that author. It was included like this in order to reduce external
// dependencies.
type UUID [16]byte

const (
	reservedNCS       byte = 0x80
	reservedRFC4122   byte = 0x40
	reservedMicrosoft byte = 0x20
	reservedFuture    byte = 0x00

	// pattern used to parse hex string representation of the UUID.
	hexPattern = `^(urn\:uuid\:)?\{?([a-z0-9]{8})-([a-z0-9]{4})-` +
		`([1-5][a-z0-9]{3})-([a-z0-9]{4})-([a-z0-9]{12})\}?$`
)

var (
	hexRX = regexp.MustCompile(hexPattern)
)

// String returns unparsed version of the generated UUID sequence.
func (u *UUID) String() string {
	return fmt.Sprintf("%x-%x-%x-%x-%x",
		u[0:4], u[4:6], u[6:8], u[8:10], u[10:])
}

// MarshalText marshals the UUID to a string.
func (u *UUID) MarshalText() ([]byte, error) {
	return []byte(u.String()), nil
}

// UnmarshalText unmarshals the UUID from a hex string to a UUID instance.
// This function accepts UUID string in following formats:
//
//     uuid.ParseHex("6ba7b814-9dad-11d1-80b4-00c04fd430c8")
//     uuid.ParseHex("{6ba7b814-9dad-11d1-80b4-00c04fd430c8}")
//     uuid.ParseHex("urn:uuid:6ba7b814-9dad-11d1-80b4-00c04fd430c8")
func (u *UUID) UnmarshalText(text []byte) error {
	md := hexRX.FindSubmatch(text)
	if len(md) == 0 {
		return goof.New("invalid uuid string")
	}
	hash := []byte{}
	for x := 2; x < 7; x++ {
		hash = append(hash, md[x]...)
	}
	if _, err := hex.Decode(u[:], hash); err != nil {
		return err
	}
	return nil
}

// NewUUID returns a new UUID.
func NewUUID() (*UUID, error) {
	u := &UUID{}
	if _, err := rand.Read(u[:]); err != nil {
		return nil, err
	}
	u.setVariant(reservedRFC4122)
	u.setVersion(4)
	return u, nil
}

// MustNewUUID is like NewUUID but panics if it encounters an error when
// creating a new UUID.
func MustNewUUID() *UUID {
	uuid, err := NewUUID()
	if err != nil {
		panic(err)
	}
	return uuid
}

// ParseUUID is a helper function on top of UnmarshalText.
func ParseUUID(s string) (*UUID, error) {
	u := &UUID{}
	if err := u.UnmarshalText([]byte(s)); err != nil {
		return nil, err
	}
	return u, nil
}

// setVariant sets the two most significant bits (bits 6 and 7) of the
// clock_seq_hi_and_reserved to zero and one, respectively.
func (u *UUID) setVariant(v byte) {
	switch v {
	case reservedNCS:
		u[8] = (u[8] | reservedNCS) & 0xBF
	case reservedRFC4122:
		u[8] = (u[8] | reservedRFC4122) & 0x7F
	case reservedMicrosoft:
		u[8] = (u[8] | reservedMicrosoft) & 0x3F
	}
}

// setVersion sets the four most significant bits (bits 12 through 15) of the
// time_hi_and_version field to the 4-bit version number.
func (u *UUID) setVersion(v byte) {
	u[6] = (u[6] & 0xF) | (v << 4)
}
