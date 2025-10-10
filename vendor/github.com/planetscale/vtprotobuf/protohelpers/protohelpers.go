// Package protohelpers provides helper functions for encoding and decoding protobuf messages.
// The spec can be found at https://protobuf.dev/programming-guides/encoding/.
package protohelpers

import (
	"fmt"
	"io"
	"math/bits"
)

var (
	// ErrInvalidLength is returned when decoding a negative length.
	ErrInvalidLength = fmt.Errorf("proto: negative length found during unmarshaling")
	// ErrIntOverflow is returned when decoding a varint representation of an integer that overflows 64 bits.
	ErrIntOverflow = fmt.Errorf("proto: integer overflow")
	// ErrUnexpectedEndOfGroup is returned when decoding a group end without a corresponding group start.
	ErrUnexpectedEndOfGroup = fmt.Errorf("proto: unexpected end of group")
)

// EncodeVarint encodes a uint64 into a varint-encoded byte slice and returns the offset of the encoded value.
// The provided offset is the offset after the last byte of the encoded value.
func EncodeVarint(dAtA []byte, offset int, v uint64) int {
	offset -= SizeOfVarint(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}

// SizeOfVarint returns the size of the varint-encoded value.
func SizeOfVarint(x uint64) (n int) {
	return (bits.Len64(x|1) + 6) / 7
}

// SizeOfZigzag returns the size of the zigzag-encoded value.
func SizeOfZigzag(x uint64) (n int) {
	return SizeOfVarint(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}

// Skip the first record of the byte slice and return the offset of the next record.
func Skip(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflow
			}
			if iNdEx >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		wireType := int(wire & 0x7)
		switch wireType {
		case 0:
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflow
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if dAtA[iNdEx-1] < 0x80 {
					break
				}
			}
		case 1:
			iNdEx += 8
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflow
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if length < 0 {
				return 0, ErrInvalidLength
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroup
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLength
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}
