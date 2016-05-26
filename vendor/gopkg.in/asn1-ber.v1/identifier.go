package ber

import (
	"errors"
	"fmt"
	"io"
	"math"
)

func readIdentifier(reader io.Reader) (Identifier, int, error) {
	identifier := Identifier{}
	read := 0

	// identifier byte
	b, err := readByte(reader)
	if err != nil {
		if Debug {
			fmt.Printf("error reading identifier byte: %v\n", err)
		}
		return Identifier{}, read, err
	}
	read++

	identifier.ClassType = Class(b) & ClassBitmask
	identifier.TagType = Type(b) & TypeBitmask

	if tag := Tag(b) & TagBitmask; tag != HighTag {
		// short-form tag
		identifier.Tag = tag
		return identifier, read, nil
	}

	// high-tag-number tag
	tagBytes := 0
	for {
		b, err := readByte(reader)
		if err != nil {
			if Debug {
				fmt.Printf("error reading high-tag-number tag byte %d: %v\n", tagBytes, err)
			}
			return Identifier{}, read, err
		}
		tagBytes++
		read++

		// Lowest 7 bits get appended to the tag value (x.690, 8.1.2.4.2.b)
		identifier.Tag <<= 7
		identifier.Tag |= Tag(b) & HighTagValueBitmask

		// First byte may not be all zeros (x.690, 8.1.2.4.2.c)
		if tagBytes == 1 && identifier.Tag == 0 {
			return Identifier{}, read, errors.New("invalid first high-tag-number tag byte")
		}
		// Overflow of int64
		// TODO: support big int tags?
		if tagBytes > 9 {
			return Identifier{}, read, errors.New("high-tag-number tag overflow")
		}

		// Top bit of 0 means this is the last byte in the high-tag-number tag (x.690, 8.1.2.4.2.a)
		if Tag(b)&HighTagContinueBitmask == 0 {
			break
		}
	}

	return identifier, read, nil
}

func encodeIdentifier(identifier Identifier) []byte {
	b := []byte{0x0}
	b[0] |= byte(identifier.ClassType)
	b[0] |= byte(identifier.TagType)

	if identifier.Tag < HighTag {
		// Short-form
		b[0] |= byte(identifier.Tag)
	} else {
		// high-tag-number
		b[0] |= byte(HighTag)

		tag := identifier.Tag

		highBit := uint(63)
		for {
			if tag&(1<<highBit) != 0 {
				break
			}
			highBit--
		}

		tagBytes := int(math.Ceil(float64(highBit) / 7.0))
		for i := tagBytes - 1; i >= 0; i-- {
			offset := uint(i) * 7
			mask := Tag(0x7f) << offset
			tagByte := (tag & mask) >> offset
			if i != 0 {
				tagByte |= 0x80
			}
			b = append(b, byte(tagByte))
		}
	}
	return b
}
