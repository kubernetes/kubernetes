// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"errors"
	"fmt"
	"reflect"
)

// SimpleValue represents CBOR simple value.
// CBOR simple value is:
//   - an extension point like CBOR tag.
//   - a subset of CBOR major type 7 that isn't floating-point.
//   - "identified by a number between 0 and 255, but distinct from that number itself".
//     For example, "a simple value 2 is not equivalent to an integer 2" as a CBOR map key.
//
// CBOR simple values identified by 20..23 are: "false", "true" , "null", and "undefined".
// Other CBOR simple values are currently unassigned/reserved by IANA.
type SimpleValue uint8

var (
	typeSimpleValue = reflect.TypeOf(SimpleValue(0))
)

// MarshalCBOR encodes SimpleValue as CBOR simple value (major type 7).
func (sv SimpleValue) MarshalCBOR() ([]byte, error) {
	// RFC 8949 3.3. Floating-Point Numbers and Values with No Content says:
	// "An encoder MUST NOT issue two-byte sequences that start with 0xf8
	// (major type 7, additional information 24) and continue with a byte
	// less than 0x20 (32 decimal). Such sequences are not well-formed.
	// (This implies that an encoder cannot encode false, true, null, or
	// undefined in two-byte sequences and that only the one-byte variants
	// of these are well-formed; more generally speaking, each simple value
	// only has a single representation variant)."

	switch {
	case sv <= maxSimpleValueInAdditionalInformation:
		return []byte{byte(cborTypePrimitives) | byte(sv)}, nil

	case sv >= minSimpleValueIn1ByteArgument:
		return []byte{byte(cborTypePrimitives) | additionalInformationWith1ByteArgument, byte(sv)}, nil

	default:
		return nil, &UnsupportedValueError{msg: fmt.Sprintf("SimpleValue(%d)", sv)}
	}
}

// UnmarshalCBOR decodes CBOR simple value (major type 7) to SimpleValue.
//
// Deprecated: No longer used by this codec; kept for compatibility
// with user apps that directly call this function.
func (sv *SimpleValue) UnmarshalCBOR(data []byte) error {
	if sv == nil {
		return errors.New("cbor.SimpleValue: UnmarshalCBOR on nil pointer")
	}

	d := decoder{data: data, dm: defaultDecMode}

	// Check well-formedness of CBOR data item.
	// SimpleValue.UnmarshalCBOR() is exported, so
	// the codec needs to support same behavior for:
	// - Unmarshal(data, *SimpleValue)
	// - SimpleValue.UnmarshalCBOR(data)
	err := d.wellformed(false, false)
	if err != nil {
		return err
	}

	return sv.unmarshalCBOR(data)
}

// unmarshalCBOR decodes CBOR simple value (major type 7) to SimpleValue.
// This function assumes data is well-formed, and does not perform bounds checking.
// This function is called by Unmarshal().
func (sv *SimpleValue) unmarshalCBOR(data []byte) error {
	if sv == nil {
		return errors.New("cbor.SimpleValue: UnmarshalCBOR on nil pointer")
	}

	d := decoder{data: data, dm: defaultDecMode}

	typ, ai, val := d.getHead()

	if typ != cborTypePrimitives {
		return &UnmarshalTypeError{CBORType: typ.String(), GoType: "SimpleValue"}
	}
	if ai > additionalInformationWith1ByteArgument {
		return &UnmarshalTypeError{CBORType: typ.String(), GoType: "SimpleValue", errorMsg: "not simple values"}
	}

	// It is safe to cast val to uint8 here because
	// - data is already verified to be well-formed CBOR simple value and
	// - val is <= math.MaxUint8.
	*sv = SimpleValue(val)
	return nil
}
