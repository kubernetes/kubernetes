// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"

	protoV2 "google.golang.org/protobuf/proto"
)

var (
	// Deprecated: No longer returned.
	ErrNil = errors.New("proto: Marshal called with nil")

	// Deprecated: No longer returned.
	ErrTooLarge = errors.New("proto: message encodes to over 2 GB")

	// Deprecated: No longer returned.
	ErrInternalBadWireType = errors.New("proto: internal error: bad wiretype for oneof")
)

// Deprecated: Do not use.
type Stats struct{ Emalloc, Dmalloc, Encode, Decode, Chit, Cmiss, Size uint64 }

// Deprecated: Do not use.
func GetStats() Stats { return Stats{} }

// Deprecated: Do not use.
func MarshalMessageSet(interface{}) ([]byte, error) {
	return nil, errors.New("proto: not implemented")
}

// Deprecated: Do not use.
func UnmarshalMessageSet([]byte, interface{}) error {
	return errors.New("proto: not implemented")
}

// Deprecated: Do not use.
func MarshalMessageSetJSON(interface{}) ([]byte, error) {
	return nil, errors.New("proto: not implemented")
}

// Deprecated: Do not use.
func UnmarshalMessageSetJSON([]byte, interface{}) error {
	return errors.New("proto: not implemented")
}

// Deprecated: Do not use.
func RegisterMessageSetType(Message, int32, string) {}

// Deprecated: Do not use.
func EnumName(m map[int32]string, v int32) string {
	s, ok := m[v]
	if ok {
		return s
	}
	return strconv.Itoa(int(v))
}

// Deprecated: Do not use.
func UnmarshalJSONEnum(m map[string]int32, data []byte, enumName string) (int32, error) {
	if data[0] == '"' {
		// New style: enums are strings.
		var repr string
		if err := json.Unmarshal(data, &repr); err != nil {
			return -1, err
		}
		val, ok := m[repr]
		if !ok {
			return 0, fmt.Errorf("unrecognized enum %s value %q", enumName, repr)
		}
		return val, nil
	}
	// Old style: enums are ints.
	var val int32
	if err := json.Unmarshal(data, &val); err != nil {
		return 0, fmt.Errorf("cannot unmarshal %#q into enum %s", data, enumName)
	}
	return val, nil
}

// Deprecated: Do not use; this type existed for intenal-use only.
type InternalMessageInfo struct{}

// Deprecated: Do not use; this method existed for intenal-use only.
func (*InternalMessageInfo) DiscardUnknown(m Message) {
	DiscardUnknown(m)
}

// Deprecated: Do not use; this method existed for intenal-use only.
func (*InternalMessageInfo) Marshal(b []byte, m Message, deterministic bool) ([]byte, error) {
	return protoV2.MarshalOptions{Deterministic: deterministic}.MarshalAppend(b, MessageV2(m))
}

// Deprecated: Do not use; this method existed for intenal-use only.
func (*InternalMessageInfo) Merge(dst, src Message) {
	protoV2.Merge(MessageV2(dst), MessageV2(src))
}

// Deprecated: Do not use; this method existed for intenal-use only.
func (*InternalMessageInfo) Size(m Message) int {
	return protoV2.Size(MessageV2(m))
}

// Deprecated: Do not use; this method existed for intenal-use only.
func (*InternalMessageInfo) Unmarshal(m Message, b []byte) error {
	return protoV2.UnmarshalOptions{Merge: true}.Unmarshal(b, MessageV2(m))
}
