package issue411

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"strconv"
)

// TraceID is a random 128bit identifier for a trace
type TraceID struct {
	Low  uint64 `json:"lo"`
	High uint64 `json:"hi"`
}

// SpanID is a random 64bit identifier for a span
type SpanID uint64

// ------- TraceID -------

// NewTraceID creates a new TraceID from two 64bit unsigned ints.
func NewTraceID(high, low uint64) TraceID {
	return TraceID{High: high, Low: low}
}

func (t TraceID) String() string {
	if t.High == 0 {
		return fmt.Sprintf("%x", t.Low)
	}
	return fmt.Sprintf("%x%016x", t.High, t.Low)
}

// TraceIDFromString creates a TraceID from a hexadecimal string
func TraceIDFromString(s string) (TraceID, error) {
	var hi, lo uint64
	var err error
	if len(s) > 32 {
		return TraceID{}, fmt.Errorf("TraceID cannot be longer than 32 hex characters: %s", s)
	} else if len(s) > 16 {
		hiLen := len(s) - 16
		if hi, err = strconv.ParseUint(s[0:hiLen], 16, 64); err != nil {
			return TraceID{}, err
		}
		if lo, err = strconv.ParseUint(s[hiLen:], 16, 64); err != nil {
			return TraceID{}, err
		}
	} else {
		if lo, err = strconv.ParseUint(s, 16, 64); err != nil {
			return TraceID{}, err
		}
	}
	return TraceID{High: hi, Low: lo}, nil
}

// MarshalText is called by encoding/json, which we do not want people to use.
func (t TraceID) MarshalText() ([]byte, error) {
	return nil, fmt.Errorf("unsupported method TraceID.MarshalText; please use github.com/gogo/protobuf/jsonpb for marshalling")
}

// UnmarshalText is called by encoding/json, which we do not want people to use.
func (t *TraceID) UnmarshalText(text []byte) error {
	return fmt.Errorf("unsupported method TraceID.UnmarshalText; please use github.com/gogo/protobuf/jsonpb for marshalling")
}

// Size returns the size of this datum in protobuf. It is always 16 bytes.
func (t *TraceID) Size() int {
	return 16
}

// Marshal converts trace ID into a binary representation. Called by protobuf serialization.
func (t TraceID) Marshal() ([]byte, error) {
	b := make([]byte, t.Size())
	_, err := t.MarshalTo(b)
	return b, err
}

// MarshalTo converts trace ID into a binary representation. Called by protobuf serialization.
func (t *TraceID) MarshalTo(data []byte) (n int, err error) {
	var b [16]byte
	binary.BigEndian.PutUint64(b[:8], uint64(t.High))
	binary.BigEndian.PutUint64(b[8:], uint64(t.Low))
	return marshalBytes(data, b[:])
}

// Unmarshal inflates this trace ID from binary representation. Called by protobuf serialization.
func (t *TraceID) Unmarshal(data []byte) error {
	if len(data) < 16 {
		return fmt.Errorf("buffer is too short")
	}
	t.High = binary.BigEndian.Uint64(data[:8])
	t.Low = binary.BigEndian.Uint64(data[8:])
	return nil
}

func marshalBytes(dst []byte, src []byte) (n int, err error) {
	if len(dst) < len(src) {
		return 0, fmt.Errorf("buffer is too short")
	}
	return copy(dst, src), nil
}

// MarshalJSON converts trace id into a base64 string enclosed in quotes.
// Used by protobuf JSON serialization.
// Example: {high:2, low:1} => "AAAAAAAAAAIAAAAAAAAAAQ==".
func (t TraceID) MarshalJSON() ([]byte, error) {
	var b [16]byte
	_, err := t.MarshalTo(b[:]) // can only error on incorrect buffer size
	if err != nil {
		return []byte{}, err
	}
	s := make([]byte, 24+2)
	base64.StdEncoding.Encode(s[1:25], b[:])
	s[0], s[25] = '"', '"'
	return s, nil
}

// UnmarshalJSON inflates trace id from base64 string, possibly enclosed in quotes.
// User by protobuf JSON serialization.
func (t *TraceID) UnmarshalJSON(data []byte) error {
	s := string(data)
	if l := len(s); l > 2 && s[0] == '"' && s[l-1] == '"' {
		s = s[1 : l-1]
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return fmt.Errorf("cannot unmarshal TraceID from string '%s': %v", string(data), err)
	}
	return t.Unmarshal(b)
}

// ------- SpanID -------

// NewSpanID creates a new SpanID from a 64bit unsigned int.
func NewSpanID(v uint64) SpanID {
	return SpanID(v)
}

func (s SpanID) String() string {
	return fmt.Sprintf("%x", uint64(s))
}

// SpanIDFromString creates a SpanID from a hexadecimal string
func SpanIDFromString(s string) (SpanID, error) {
	if len(s) > 16 {
		return SpanID(0), fmt.Errorf("SpanID cannot be longer than 16 hex characters: %s", s)
	}
	id, err := strconv.ParseUint(s, 16, 64)
	if err != nil {
		return SpanID(0), err
	}
	return SpanID(id), nil
}

// MarshalText is called by encoding/json, which we do not want people to use.
func (s SpanID) MarshalText() ([]byte, error) {
	return nil, fmt.Errorf("unsupported method SpanID.MarshalText; please use github.com/gogo/protobuf/jsonpb for marshalling")
}

// UnmarshalText is called by encoding/json, which we do not want people to use.
func (s *SpanID) UnmarshalText(text []byte) error {
	return fmt.Errorf("unsupported method SpanID.UnmarshalText; please use github.com/gogo/protobuf/jsonpb for marshalling")
}

// Size returns the size of this datum in protobuf. It is always 8 bytes.
func (s *SpanID) Size() int {
	return 8
}

// Marshal converts span ID into a binary representation. Called by protobuf serialization.
func (s SpanID) Marshal() ([]byte, error) {
	b := make([]byte, s.Size())
	_, err := s.MarshalTo(b)
	return b, err
}

// MarshalTo converts span ID into a binary representation. Called by protobuf serialization.
func (s *SpanID) MarshalTo(data []byte) (n int, err error) {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], uint64(*s))
	return marshalBytes(data, b[:])
}

// Unmarshal inflates span ID from a binary representation. Called by protobuf serialization.
func (s *SpanID) Unmarshal(data []byte) error {
	if len(data) < 8 {
		return fmt.Errorf("buffer is too short")
	}
	*s = NewSpanID(binary.BigEndian.Uint64(data))
	return nil
}

// MarshalJSON converts span id into a base64 string enclosed in quotes.
// Used by protobuf JSON serialization.
// Example: {1} => "AAAAAAAAAAE=".
func (s SpanID) MarshalJSON() ([]byte, error) {
	var b [8]byte
	_, err := s.MarshalTo(b[:]) // can only error on incorrect buffer size
	if err != nil {
		return []byte{}, err
	}
	v := make([]byte, 12+2)
	base64.StdEncoding.Encode(v[1:13], b[:])
	v[0], v[13] = '"', '"'
	return v, nil
}

// UnmarshalJSON inflates span id from base64 string, possibly enclosed in quotes.
// User by protobuf JSON serialization.
//
// There appears to be a bug in gogoproto, as this function is only called for numeric values.
// https://github.com/gogo/protobuf/issues/411#issuecomment-393856837
func (s *SpanID) UnmarshalJSON(data []byte) error {
	str := string(data)
	if l := len(str); l > 2 && str[0] == '"' && str[l-1] == '"' {
		str = str[1 : l-1]
	}
	b, err := base64.StdEncoding.DecodeString(str)
	if err != nil {
		return fmt.Errorf("cannot unmarshal SpanID from string '%s': %v", string(data), err)
	}
	return s.Unmarshal(b)
}
