// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"encoding/binary"
	"errors"
	"io"
	"math"
	"strconv"

	"github.com/x448/float16"
)

// SyntaxError is a description of a CBOR syntax error.
type SyntaxError struct {
	msg string
}

func (e *SyntaxError) Error() string { return e.msg }

// SemanticError is a description of a CBOR semantic error.
type SemanticError struct {
	msg string
}

func (e *SemanticError) Error() string { return e.msg }

// MaxNestedLevelError indicates exceeded max nested level of any combination of CBOR arrays/maps/tags.
type MaxNestedLevelError struct {
	maxNestedLevels int
}

func (e *MaxNestedLevelError) Error() string {
	return "cbor: exceeded max nested level " + strconv.Itoa(e.maxNestedLevels)
}

// MaxArrayElementsError indicates exceeded max number of elements for CBOR arrays.
type MaxArrayElementsError struct {
	maxArrayElements int
}

func (e *MaxArrayElementsError) Error() string {
	return "cbor: exceeded max number of elements " + strconv.Itoa(e.maxArrayElements) + " for CBOR array"
}

// MaxMapPairsError indicates exceeded max number of key-value pairs for CBOR maps.
type MaxMapPairsError struct {
	maxMapPairs int
}

func (e *MaxMapPairsError) Error() string {
	return "cbor: exceeded max number of key-value pairs " + strconv.Itoa(e.maxMapPairs) + " for CBOR map"
}

// IndefiniteLengthError indicates found disallowed indefinite length items.
type IndefiniteLengthError struct {
	t cborType
}

func (e *IndefiniteLengthError) Error() string {
	return "cbor: indefinite-length " + e.t.String() + " isn't allowed"
}

// TagsMdError indicates found disallowed CBOR tags.
type TagsMdError struct {
}

func (e *TagsMdError) Error() string {
	return "cbor: CBOR tag isn't allowed"
}

// ExtraneousDataError indicates found extraneous data following well-formed CBOR data item.
type ExtraneousDataError struct {
	numOfBytes int // number of bytes of extraneous data
	index      int // location of extraneous data
}

func (e *ExtraneousDataError) Error() string {
	return "cbor: " + strconv.Itoa(e.numOfBytes) + " bytes of extraneous data starting at index " + strconv.Itoa(e.index)
}

// wellformed checks whether the CBOR data item is well-formed.
// allowExtraData indicates if extraneous data is allowed after the CBOR data item.
// - use allowExtraData = true when using Decoder.Decode()
// - use allowExtraData = false when using Unmarshal()
func (d *decoder) wellformed(allowExtraData bool, checkBuiltinTags bool) error {
	if len(d.data) == d.off {
		return io.EOF
	}
	_, err := d.wellformedInternal(0, checkBuiltinTags)
	if err == nil {
		if !allowExtraData && d.off != len(d.data) {
			err = &ExtraneousDataError{len(d.data) - d.off, d.off}
		}
	}
	return err
}

// wellformedInternal checks data's well-formedness and returns max depth and error.
func (d *decoder) wellformedInternal(depth int, checkBuiltinTags bool) (int, error) { //nolint:gocyclo
	t, _, val, indefiniteLength, err := d.wellformedHeadWithIndefiniteLengthFlag()
	if err != nil {
		return 0, err
	}

	switch t {
	case cborTypeByteString, cborTypeTextString:
		if indefiniteLength {
			if d.dm.indefLength == IndefLengthForbidden {
				return 0, &IndefiniteLengthError{t}
			}
			return d.wellformedIndefiniteString(t, depth, checkBuiltinTags)
		}
		valInt := int(val)
		if valInt < 0 {
			// Detect integer overflow
			return 0, errors.New("cbor: " + t.String() + " length " + strconv.FormatUint(val, 10) + " is too large, causing integer overflow")
		}
		if len(d.data)-d.off < valInt { // valInt+off may overflow integer
			return 0, io.ErrUnexpectedEOF
		}
		d.off += valInt

	case cborTypeArray, cborTypeMap:
		depth++
		if depth > d.dm.maxNestedLevels {
			return 0, &MaxNestedLevelError{d.dm.maxNestedLevels}
		}

		if indefiniteLength {
			if d.dm.indefLength == IndefLengthForbidden {
				return 0, &IndefiniteLengthError{t}
			}
			return d.wellformedIndefiniteArrayOrMap(t, depth, checkBuiltinTags)
		}

		valInt := int(val)
		if valInt < 0 {
			// Detect integer overflow
			return 0, errors.New("cbor: " + t.String() + " length " + strconv.FormatUint(val, 10) + " is too large, it would cause integer overflow")
		}

		if t == cborTypeArray {
			if valInt > d.dm.maxArrayElements {
				return 0, &MaxArrayElementsError{d.dm.maxArrayElements}
			}
		} else {
			if valInt > d.dm.maxMapPairs {
				return 0, &MaxMapPairsError{d.dm.maxMapPairs}
			}
		}

		count := 1
		if t == cborTypeMap {
			count = 2
		}
		maxDepth := depth
		for j := 0; j < count; j++ {
			for i := 0; i < valInt; i++ {
				var dpt int
				if dpt, err = d.wellformedInternal(depth, checkBuiltinTags); err != nil {
					return 0, err
				}
				if dpt > maxDepth {
					maxDepth = dpt // Save max depth
				}
			}
		}
		depth = maxDepth

	case cborTypeTag:
		if d.dm.tagsMd == TagsForbidden {
			return 0, &TagsMdError{}
		}

		tagNum := val

		// Scan nested tag numbers to avoid recursion.
		for {
			if len(d.data) == d.off { // Tag number must be followed by tag content.
				return 0, io.ErrUnexpectedEOF
			}
			if checkBuiltinTags {
				err = validBuiltinTag(tagNum, d.data[d.off])
				if err != nil {
					return 0, err
				}
			}
			if d.dm.bignumTag == BignumTagForbidden && (tagNum == 2 || tagNum == 3) {
				return 0, &UnacceptableDataItemError{
					CBORType: cborTypeTag.String(),
					Message:  "bignum",
				}
			}
			if getType(d.data[d.off]) != cborTypeTag {
				break
			}
			if _, _, tagNum, err = d.wellformedHead(); err != nil {
				return 0, err
			}
			depth++
			if depth > d.dm.maxNestedLevels {
				return 0, &MaxNestedLevelError{d.dm.maxNestedLevels}
			}
		}
		// Check tag content.
		return d.wellformedInternal(depth, checkBuiltinTags)
	}

	return depth, nil
}

// wellformedIndefiniteString checks indefinite length byte/text string's well-formedness and returns max depth and error.
func (d *decoder) wellformedIndefiniteString(t cborType, depth int, checkBuiltinTags bool) (int, error) {
	var err error
	for {
		if len(d.data) == d.off {
			return 0, io.ErrUnexpectedEOF
		}
		if isBreakFlag(d.data[d.off]) {
			d.off++
			break
		}
		// Peek ahead to get next type and indefinite length status.
		nt, ai := parseInitialByte(d.data[d.off])
		if t != nt {
			return 0, &SyntaxError{"cbor: wrong element type " + nt.String() + " for indefinite-length " + t.String()}
		}
		if additionalInformation(ai).isIndefiniteLength() {
			return 0, &SyntaxError{"cbor: indefinite-length " + t.String() + " chunk is not definite-length"}
		}
		if depth, err = d.wellformedInternal(depth, checkBuiltinTags); err != nil {
			return 0, err
		}
	}
	return depth, nil
}

// wellformedIndefiniteArrayOrMap checks indefinite length array/map's well-formedness and returns max depth and error.
func (d *decoder) wellformedIndefiniteArrayOrMap(t cborType, depth int, checkBuiltinTags bool) (int, error) {
	var err error
	maxDepth := depth
	i := 0
	for {
		if len(d.data) == d.off {
			return 0, io.ErrUnexpectedEOF
		}
		if isBreakFlag(d.data[d.off]) {
			d.off++
			break
		}
		var dpt int
		if dpt, err = d.wellformedInternal(depth, checkBuiltinTags); err != nil {
			return 0, err
		}
		if dpt > maxDepth {
			maxDepth = dpt
		}
		i++
		if t == cborTypeArray {
			if i > d.dm.maxArrayElements {
				return 0, &MaxArrayElementsError{d.dm.maxArrayElements}
			}
		} else {
			if i%2 == 0 && i/2 > d.dm.maxMapPairs {
				return 0, &MaxMapPairsError{d.dm.maxMapPairs}
			}
		}
	}
	if t == cborTypeMap && i%2 == 1 {
		return 0, &SyntaxError{"cbor: unexpected \"break\" code"}
	}
	return maxDepth, nil
}

func (d *decoder) wellformedHeadWithIndefiniteLengthFlag() (
	t cborType,
	ai byte,
	val uint64,
	indefiniteLength bool,
	err error,
) {
	t, ai, val, err = d.wellformedHead()
	if err != nil {
		return
	}
	indefiniteLength = additionalInformation(ai).isIndefiniteLength()
	return
}

func (d *decoder) wellformedHead() (t cborType, ai byte, val uint64, err error) {
	dataLen := len(d.data) - d.off
	if dataLen == 0 {
		return 0, 0, 0, io.ErrUnexpectedEOF
	}

	t, ai = parseInitialByte(d.data[d.off])
	val = uint64(ai)
	d.off++
	dataLen--

	if ai <= maxAdditionalInformationWithoutArgument {
		return t, ai, val, nil
	}

	if ai == additionalInformationWith1ByteArgument {
		const argumentSize = 1
		if dataLen < argumentSize {
			return 0, 0, 0, io.ErrUnexpectedEOF
		}
		val = uint64(d.data[d.off])
		d.off++
		if t == cborTypePrimitives && val < 32 {
			return 0, 0, 0, &SyntaxError{"cbor: invalid simple value " + strconv.Itoa(int(val)) + " for type " + t.String()}
		}
		return t, ai, val, nil
	}

	if ai == additionalInformationWith2ByteArgument {
		const argumentSize = 2
		if dataLen < argumentSize {
			return 0, 0, 0, io.ErrUnexpectedEOF
		}
		val = uint64(binary.BigEndian.Uint16(d.data[d.off : d.off+argumentSize]))
		d.off += argumentSize
		if t == cborTypePrimitives {
			if err := d.acceptableFloat(float64(float16.Frombits(uint16(val)).Float32())); err != nil {
				return 0, 0, 0, err
			}
		}
		return t, ai, val, nil
	}

	if ai == additionalInformationWith4ByteArgument {
		const argumentSize = 4
		if dataLen < argumentSize {
			return 0, 0, 0, io.ErrUnexpectedEOF
		}
		val = uint64(binary.BigEndian.Uint32(d.data[d.off : d.off+argumentSize]))
		d.off += argumentSize
		if t == cborTypePrimitives {
			if err := d.acceptableFloat(float64(math.Float32frombits(uint32(val)))); err != nil {
				return 0, 0, 0, err
			}
		}
		return t, ai, val, nil
	}

	if ai == additionalInformationWith8ByteArgument {
		const argumentSize = 8
		if dataLen < argumentSize {
			return 0, 0, 0, io.ErrUnexpectedEOF
		}
		val = binary.BigEndian.Uint64(d.data[d.off : d.off+argumentSize])
		d.off += argumentSize
		if t == cborTypePrimitives {
			if err := d.acceptableFloat(math.Float64frombits(val)); err != nil {
				return 0, 0, 0, err
			}
		}
		return t, ai, val, nil
	}

	if additionalInformation(ai).isIndefiniteLength() {
		switch t {
		case cborTypePositiveInt, cborTypeNegativeInt, cborTypeTag:
			return 0, 0, 0, &SyntaxError{"cbor: invalid additional information " + strconv.Itoa(int(ai)) + " for type " + t.String()}
		case cborTypePrimitives: // 0xff (break code) should not be outside wellformedIndefinite().
			return 0, 0, 0, &SyntaxError{"cbor: unexpected \"break\" code"}
		}
		return t, ai, val, nil
	}

	// ai == 28, 29, 30
	return 0, 0, 0, &SyntaxError{"cbor: invalid additional information " + strconv.Itoa(int(ai)) + " for type " + t.String()}
}

func (d *decoder) acceptableFloat(f float64) error {
	switch {
	case d.dm.nanDec == NaNDecodeForbidden && math.IsNaN(f):
		return &UnacceptableDataItemError{
			CBORType: cborTypePrimitives.String(),
			Message:  "floating-point NaN",
		}
	case d.dm.infDec == InfDecodeForbidden && math.IsInf(f, 0):
		return &UnacceptableDataItemError{
			CBORType: cborTypePrimitives.String(),
			Message:  "floating-point infinity",
		}
	}
	return nil
}
