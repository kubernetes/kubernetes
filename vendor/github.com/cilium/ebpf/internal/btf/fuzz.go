//go:build gofuzz
// +build gofuzz

// Use with https://github.com/dvyukov/go-fuzz

package btf

import (
	"bytes"
	"encoding/binary"

	"github.com/cilium/ebpf/internal"
)

func FuzzSpec(data []byte) int {
	if len(data) < binary.Size(btfHeader{}) {
		return -1
	}

	spec, err := loadNakedSpec(bytes.NewReader(data), internal.NativeEndian, nil, nil)
	if err != nil {
		if spec != nil {
			panic("spec is not nil")
		}
		return 0
	}
	if spec == nil {
		panic("spec is nil")
	}
	return 1
}

func FuzzExtInfo(data []byte) int {
	if len(data) < binary.Size(btfExtHeader{}) {
		return -1
	}

	table := stringTable("\x00foo\x00barfoo\x00")
	info, err := parseExtInfo(bytes.NewReader(data), internal.NativeEndian, table)
	if err != nil {
		if info != nil {
			panic("info is not nil")
		}
		return 0
	}
	if info == nil {
		panic("info is nil")
	}
	return 1
}
