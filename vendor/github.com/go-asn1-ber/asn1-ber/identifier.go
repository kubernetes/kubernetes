package ber

import (
	"errors"
	"fmt"
	"io"
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
			return Identifier{}, read, unexpectedEOF(err)
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

		b = append(b, encodeHighTag(tag)...)
	}
	return b
}

func encodeHighTag(tag Tag) []byte {
	// set cap=4 to hopefully avoid additional allocations
	b := make([]byte, 0, 4)
	for tag != 0 {
		// t := last 7 bits of tag (HighTagValueBitmask = 0x7F)
		t := tag & HighTagValueBitmask

		// right shift tag 7 to remove what was just pulled off
		tag >>= 7

		// if b already has entries this entry needs a continuation bit (0x80)
		if len(b) != 0 {
			t |= HighTagContinueBitmask
		}

		b = append(b, byte(t))
	}
	// reverse
	// since bits were pulled off 'tag' small to high the byte slice is in reverse order.
	// example: tag = 0xFF results in {0x7F, 0x01 + 0x80 (continuation bit)}
	// this needs to be reversed into 0x81 0x7F
	for i, j := 0, len(b)-1; i < len(b)/2; i++ {
		b[i], b[j-i] = b[j-i], b[i]
	}
	return b
}
