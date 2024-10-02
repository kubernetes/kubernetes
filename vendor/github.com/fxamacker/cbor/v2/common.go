// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"fmt"
	"strconv"
)

type cborType uint8

const (
	cborTypePositiveInt cborType = 0x00
	cborTypeNegativeInt cborType = 0x20
	cborTypeByteString  cborType = 0x40
	cborTypeTextString  cborType = 0x60
	cborTypeArray       cborType = 0x80
	cborTypeMap         cborType = 0xa0
	cborTypeTag         cborType = 0xc0
	cborTypePrimitives  cborType = 0xe0
)

func (t cborType) String() string {
	switch t {
	case cborTypePositiveInt:
		return "positive integer"
	case cborTypeNegativeInt:
		return "negative integer"
	case cborTypeByteString:
		return "byte string"
	case cborTypeTextString:
		return "UTF-8 text string"
	case cborTypeArray:
		return "array"
	case cborTypeMap:
		return "map"
	case cborTypeTag:
		return "tag"
	case cborTypePrimitives:
		return "primitives"
	default:
		return "Invalid type " + strconv.Itoa(int(t))
	}
}

type additionalInformation uint8

const (
	maxAdditionalInformationWithoutArgument = 23
	additionalInformationWith1ByteArgument  = 24
	additionalInformationWith2ByteArgument  = 25
	additionalInformationWith4ByteArgument  = 26
	additionalInformationWith8ByteArgument  = 27

	// For major type 7.
	additionalInformationAsFalse     = 20
	additionalInformationAsTrue      = 21
	additionalInformationAsNull      = 22
	additionalInformationAsUndefined = 23
	additionalInformationAsFloat16   = 25
	additionalInformationAsFloat32   = 26
	additionalInformationAsFloat64   = 27

	// For major type 2, 3, 4, 5.
	additionalInformationAsIndefiniteLengthFlag = 31
)

const (
	maxSimpleValueInAdditionalInformation = 23
	minSimpleValueIn1ByteArgument         = 32
)

func (ai additionalInformation) isIndefiniteLength() bool {
	return ai == additionalInformationAsIndefiniteLengthFlag
}

const (
	// From RFC 8949 Section 3:
	//   "The initial byte of each encoded data item contains both information about the major type
	//   (the high-order 3 bits, described in Section 3.1) and additional information
	//   (the low-order 5 bits)."

	// typeMask is used to extract major type in initial byte of encoded data item.
	typeMask = 0xe0

	// additionalInformationMask is used to extract additional information in initial byte of encoded data item.
	additionalInformationMask = 0x1f
)

func getType(raw byte) cborType {
	return cborType(raw & typeMask)
}

func getAdditionalInformation(raw byte) byte {
	return raw & additionalInformationMask
}

func isBreakFlag(raw byte) bool {
	return raw == cborBreakFlag
}

func parseInitialByte(b byte) (t cborType, ai byte) {
	return getType(b), getAdditionalInformation(b)
}

const (
	tagNumRFC3339Time                    = 0
	tagNumEpochTime                      = 1
	tagNumUnsignedBignum                 = 2
	tagNumNegativeBignum                 = 3
	tagNumExpectedLaterEncodingBase64URL = 21
	tagNumExpectedLaterEncodingBase64    = 22
	tagNumExpectedLaterEncodingBase16    = 23
	tagNumSelfDescribedCBOR              = 55799
)

const (
	cborBreakFlag                          = byte(0xff)
	cborByteStringWithIndefiniteLengthHead = byte(0x5f)
	cborTextStringWithIndefiniteLengthHead = byte(0x7f)
	cborArrayWithIndefiniteLengthHead      = byte(0x9f)
	cborMapWithIndefiniteLengthHead        = byte(0xbf)
)

var (
	cborFalse            = []byte{0xf4}
	cborTrue             = []byte{0xf5}
	cborNil              = []byte{0xf6}
	cborNaN              = []byte{0xf9, 0x7e, 0x00}
	cborPositiveInfinity = []byte{0xf9, 0x7c, 0x00}
	cborNegativeInfinity = []byte{0xf9, 0xfc, 0x00}
)

// validBuiltinTag checks that supported built-in tag numbers are followed by expected content types.
func validBuiltinTag(tagNum uint64, contentHead byte) error {
	t := getType(contentHead)
	switch tagNum {
	case tagNumRFC3339Time:
		// Tag content (date/time text string in RFC 3339 format) must be string type.
		if t != cborTypeTextString {
			return newInadmissibleTagContentTypeError(
				tagNumRFC3339Time,
				"text string",
				t.String())
		}
		return nil

	case tagNumEpochTime:
		// Tag content (epoch date/time) must be uint, int, or float type.
		if t != cborTypePositiveInt && t != cborTypeNegativeInt && (contentHead < 0xf9 || contentHead > 0xfb) {
			return newInadmissibleTagContentTypeError(
				tagNumEpochTime,
				"integer or floating-point number",
				t.String())
		}
		return nil

	case tagNumUnsignedBignum, tagNumNegativeBignum:
		// Tag content (bignum) must be byte type.
		if t != cborTypeByteString {
			return newInadmissibleTagContentTypeErrorf(
				fmt.Sprintf(
					"tag number %d or %d must be followed by byte string, got %s",
					tagNumUnsignedBignum,
					tagNumNegativeBignum,
					t.String(),
				))
		}
		return nil

	case tagNumExpectedLaterEncodingBase64URL, tagNumExpectedLaterEncodingBase64, tagNumExpectedLaterEncodingBase16:
		// From RFC 8949 3.4.5.2:
		//   The data item tagged can be a byte string or any other data item. In the latter
		//   case, the tag applies to all of the byte string data items contained in the data
		//   item, except for those contained in a nested data item tagged with an expected
		//   conversion.
		return nil
	}

	return nil
}
