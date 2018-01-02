package test

import (
	"encoding"
	"strings"
)

type keyType struct {
	X string
}

func (k keyType) MarshalText() ([]byte, error) {
	return []byte("MANUAL__" + k.X), nil
}

func (k *keyType) UnmarshalText(text []byte) error {
	k.X = strings.TrimPrefix(string(text), "MANUAL__")
	return nil
}

var _ encoding.TextMarshaler = keyType{}
var _ encoding.TextUnmarshaler = &keyType{}

type typeForTest map[keyType]string
