// Type-Length-Value splitting and parsing for proxy protocol V2
// See spec https://www.haproxy.org/download/1.8/doc/proxy-protocol.txt sections 2.2 to 2.7 and

package proxyproto

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

const (
	// Section 2.2
	PP2_TYPE_ALPN           PP2Type = 0x01
	PP2_TYPE_AUTHORITY      PP2Type = 0x02
	PP2_TYPE_CRC32C         PP2Type = 0x03
	PP2_TYPE_NOOP           PP2Type = 0x04
	PP2_TYPE_UNIQUE_ID      PP2Type = 0x05
	PP2_TYPE_SSL            PP2Type = 0x20
	PP2_SUBTYPE_SSL_VERSION PP2Type = 0x21
	PP2_SUBTYPE_SSL_CN      PP2Type = 0x22
	PP2_SUBTYPE_SSL_CIPHER  PP2Type = 0x23
	PP2_SUBTYPE_SSL_SIG_ALG PP2Type = 0x24
	PP2_SUBTYPE_SSL_KEY_ALG PP2Type = 0x25
	PP2_TYPE_NETNS          PP2Type = 0x30

	// Section 2.2.7, reserved types
	PP2_TYPE_MIN_CUSTOM     PP2Type = 0xE0
	PP2_TYPE_MAX_CUSTOM     PP2Type = 0xEF
	PP2_TYPE_MIN_EXPERIMENT PP2Type = 0xF0
	PP2_TYPE_MAX_EXPERIMENT PP2Type = 0xF7
	PP2_TYPE_MIN_FUTURE     PP2Type = 0xF8
	PP2_TYPE_MAX_FUTURE     PP2Type = 0xFF
)

var (
	ErrTruncatedTLV    = errors.New("proxyproto: truncated TLV")
	ErrMalformedTLV    = errors.New("proxyproto: malformed TLV Value")
	ErrIncompatibleTLV = errors.New("proxyproto: incompatible TLV type")
)

// PP2Type is the proxy protocol v2 type
type PP2Type byte

// TLV is a uninterpreted Type-Length-Value for V2 protocol, see section 2.2
type TLV struct {
	Type  PP2Type
	Value []byte
}

// SplitTLVs splits the Type-Length-Value vector, returns the vector or an error.
func SplitTLVs(raw []byte) ([]TLV, error) {
	var tlvs []TLV
	for i := 0; i < len(raw); {
		tlv := TLV{
			Type: PP2Type(raw[i]),
		}
		if len(raw)-i <= 2 {
			return nil, ErrTruncatedTLV
		}
		tlvLen := int(binary.BigEndian.Uint16(raw[i+1 : i+3])) // Max length = 65K
		i += 3
		if i+tlvLen > len(raw) {
			return nil, ErrTruncatedTLV
		}
		// Ignore no-op padding
		if tlv.Type != PP2_TYPE_NOOP {
			tlv.Value = make([]byte, tlvLen)
			copy(tlv.Value, raw[i:i+tlvLen])
		}
		i += tlvLen
		tlvs = append(tlvs, tlv)
	}
	return tlvs, nil
}

// JoinTLVs joins multiple Type-Length-Value records.
func JoinTLVs(tlvs []TLV) ([]byte, error) {
	var raw []byte
	for _, tlv := range tlvs {
		if len(tlv.Value) > math.MaxUint16 {
			return nil, fmt.Errorf("proxyproto: cannot format TLV %v with length %d", tlv.Type, len(tlv.Value))
		}
		var length [2]byte
		binary.BigEndian.PutUint16(length[:], uint16(len(tlv.Value)))
		raw = append(raw, byte(tlv.Type))
		raw = append(raw, length[:]...)
		raw = append(raw, tlv.Value...)
	}
	return raw, nil
}

// Registered is true if the type is registered in the spec, see section 2.2
func (p PP2Type) Registered() bool {
	switch p {
	case PP2_TYPE_ALPN,
		PP2_TYPE_AUTHORITY,
		PP2_TYPE_CRC32C,
		PP2_TYPE_NOOP,
		PP2_TYPE_UNIQUE_ID,
		PP2_TYPE_SSL,
		PP2_SUBTYPE_SSL_VERSION,
		PP2_SUBTYPE_SSL_CN,
		PP2_SUBTYPE_SSL_CIPHER,
		PP2_SUBTYPE_SSL_SIG_ALG,
		PP2_SUBTYPE_SSL_KEY_ALG,
		PP2_TYPE_NETNS:
		return true
	}
	return false
}

// App is true if the type is reserved for application specific data, see section 2.2.7
func (p PP2Type) App() bool {
	return p >= PP2_TYPE_MIN_CUSTOM && p <= PP2_TYPE_MAX_CUSTOM
}

// Experiment is true if the type is reserved for temporary experimental use by application developers, see section 2.2.7
func (p PP2Type) Experiment() bool {
	return p >= PP2_TYPE_MIN_EXPERIMENT && p <= PP2_TYPE_MAX_EXPERIMENT
}

// Future is true is the type is reserved for future use, see section 2.2.7
func (p PP2Type) Future() bool {
	return p >= PP2_TYPE_MIN_FUTURE
}

// Spec is true if the type is covered by the spec, see section 2.2 and 2.2.7
func (p PP2Type) Spec() bool {
	return p.Registered() || p.App() || p.Experiment() || p.Future()
}
