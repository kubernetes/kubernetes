// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"errors"
)

// ByteString represents CBOR byte string (major type 2). ByteString can be used
// when using a Go []byte is not possible or convenient. For example, Go doesn't
// allow []byte as map key, so ByteString can be used to support data formats
// having CBOR map with byte string keys. ByteString can also be used to
// encode invalid UTF-8 string as CBOR byte string.
// See DecOption.MapKeyByteStringMode for more details.
type ByteString string

// Bytes returns bytes representing ByteString.
func (bs ByteString) Bytes() []byte {
	return []byte(bs)
}

// MarshalCBOR encodes ByteString as CBOR byte string (major type 2).
func (bs ByteString) MarshalCBOR() ([]byte, error) {
	e := getEncoderBuffer()
	defer putEncoderBuffer(e)

	// Encode length
	encodeHead(e, byte(cborTypeByteString), uint64(len(bs)))

	// Encode data
	buf := make([]byte, e.Len()+len(bs))
	n := copy(buf, e.Bytes())
	copy(buf[n:], bs)

	return buf, nil
}

// UnmarshalCBOR decodes CBOR byte string (major type 2) to ByteString.
// Decoding CBOR null and CBOR undefined sets ByteString to be empty.
func (bs *ByteString) UnmarshalCBOR(data []byte) error {
	if bs == nil {
		return errors.New("cbor.ByteString: UnmarshalCBOR on nil pointer")
	}

	// Decoding CBOR null and CBOR undefined to ByteString resets data.
	// This behavior is similar to decoding CBOR null and CBOR undefined to []byte.
	if len(data) == 1 && (data[0] == 0xf6 || data[0] == 0xf7) {
		*bs = ""
		return nil
	}

	d := decoder{data: data, dm: defaultDecMode}

	// Check if CBOR data type is byte string
	if typ := d.nextCBORType(); typ != cborTypeByteString {
		return &UnmarshalTypeError{CBORType: typ.String(), GoType: typeByteString.String()}
	}

	*bs = ByteString(d.parseByteString())
	return nil
}
