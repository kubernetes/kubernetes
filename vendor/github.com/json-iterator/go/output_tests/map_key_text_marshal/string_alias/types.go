package test

import (
	"encoding"
	"strings"
)

type keyType string

func (k keyType) MarshalText() ([]byte, error) {
	return []byte("MANUAL__" + k), nil
}

func (k *keyType) UnmarshalText(text []byte) error {
	*k = keyType(strings.TrimPrefix(string(text), "MANUAL__"))
	return nil
}

var _ encoding.TextMarshaler = keyType("")
var _ encoding.TextUnmarshaler = new(keyType)

type typeForTest map[keyType]string
