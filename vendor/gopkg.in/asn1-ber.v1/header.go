package ber

import (
	"errors"
	"fmt"
	"io"
)

func readHeader(reader io.Reader) (identifier Identifier, length int, read int, err error) {
	if i, c, err := readIdentifier(reader); err != nil {
		return Identifier{}, 0, read, err
	} else {
		identifier = i
		read += c
	}

	if l, c, err := readLength(reader); err != nil {
		return Identifier{}, 0, read, err
	} else {
		length = l
		read += c
	}

	// Validate length type with identifier (x.600, 8.1.3.2.a)
	if length == LengthIndefinite && identifier.TagType == TypePrimitive {
		return Identifier{}, 0, read, errors.New("indefinite length used with primitive type")
	}

	if length < LengthIndefinite {
		err = fmt.Errorf("length cannot be less than %d", LengthIndefinite)
		return
	}

	return identifier, length, read, nil
}
