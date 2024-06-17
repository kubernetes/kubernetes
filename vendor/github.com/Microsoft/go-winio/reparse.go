//go:build windows
// +build windows

package winio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strings"
	"unicode/utf16"
	"unsafe"
)

const (
	reparseTagMountPoint = 0xA0000003
	reparseTagSymlink    = 0xA000000C
)

type reparseDataBuffer struct {
	ReparseTag           uint32
	ReparseDataLength    uint16
	Reserved             uint16
	SubstituteNameOffset uint16
	SubstituteNameLength uint16
	PrintNameOffset      uint16
	PrintNameLength      uint16
}

// ReparsePoint describes a Win32 symlink or mount point.
type ReparsePoint struct {
	Target       string
	IsMountPoint bool
}

// UnsupportedReparsePointError is returned when trying to decode a non-symlink or
// mount point reparse point.
type UnsupportedReparsePointError struct {
	Tag uint32
}

func (e *UnsupportedReparsePointError) Error() string {
	return fmt.Sprintf("unsupported reparse point %x", e.Tag)
}

// DecodeReparsePoint decodes a Win32 REPARSE_DATA_BUFFER structure containing either a symlink
// or a mount point.
func DecodeReparsePoint(b []byte) (*ReparsePoint, error) {
	tag := binary.LittleEndian.Uint32(b[0:4])
	return DecodeReparsePointData(tag, b[8:])
}

func DecodeReparsePointData(tag uint32, b []byte) (*ReparsePoint, error) {
	isMountPoint := false
	switch tag {
	case reparseTagMountPoint:
		isMountPoint = true
	case reparseTagSymlink:
	default:
		return nil, &UnsupportedReparsePointError{tag}
	}
	nameOffset := 8 + binary.LittleEndian.Uint16(b[4:6])
	if !isMountPoint {
		nameOffset += 4
	}
	nameLength := binary.LittleEndian.Uint16(b[6:8])
	name := make([]uint16, nameLength/2)
	err := binary.Read(bytes.NewReader(b[nameOffset:nameOffset+nameLength]), binary.LittleEndian, &name)
	if err != nil {
		return nil, err
	}
	return &ReparsePoint{string(utf16.Decode(name)), isMountPoint}, nil
}

func isDriveLetter(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

// EncodeReparsePoint encodes a Win32 REPARSE_DATA_BUFFER structure describing a symlink or
// mount point.
func EncodeReparsePoint(rp *ReparsePoint) []byte {
	// Generate an NT path and determine if this is a relative path.
	var ntTarget string
	relative := false
	if strings.HasPrefix(rp.Target, `\\?\`) {
		ntTarget = `\??\` + rp.Target[4:]
	} else if strings.HasPrefix(rp.Target, `\\`) {
		ntTarget = `\??\UNC\` + rp.Target[2:]
	} else if len(rp.Target) >= 2 && isDriveLetter(rp.Target[0]) && rp.Target[1] == ':' {
		ntTarget = `\??\` + rp.Target
	} else {
		ntTarget = rp.Target
		relative = true
	}

	// The paths must be NUL-terminated even though they are counted strings.
	target16 := utf16.Encode([]rune(rp.Target + "\x00"))
	ntTarget16 := utf16.Encode([]rune(ntTarget + "\x00"))

	size := int(unsafe.Sizeof(reparseDataBuffer{})) - 8
	size += len(ntTarget16)*2 + len(target16)*2

	tag := uint32(reparseTagMountPoint)
	if !rp.IsMountPoint {
		tag = reparseTagSymlink
		size += 4 // Add room for symlink flags
	}

	data := reparseDataBuffer{
		ReparseTag:           tag,
		ReparseDataLength:    uint16(size),
		SubstituteNameOffset: 0,
		SubstituteNameLength: uint16((len(ntTarget16) - 1) * 2),
		PrintNameOffset:      uint16(len(ntTarget16) * 2),
		PrintNameLength:      uint16((len(target16) - 1) * 2),
	}

	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, &data)
	if !rp.IsMountPoint {
		flags := uint32(0)
		if relative {
			flags |= 1
		}
		_ = binary.Write(&b, binary.LittleEndian, flags)
	}

	_ = binary.Write(&b, binary.LittleEndian, ntTarget16)
	_ = binary.Write(&b, binary.LittleEndian, target16)
	return b.Bytes()
}
