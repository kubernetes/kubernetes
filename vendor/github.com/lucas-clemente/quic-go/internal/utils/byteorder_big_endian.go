package utils

import (
	"bytes"
	"io"
)

// BigEndian is the big-endian implementation of ByteOrder.
var BigEndian ByteOrder = bigEndian{}

type bigEndian struct{}

var _ ByteOrder = &bigEndian{}

// ReadUintN reads N bytes
func (bigEndian) ReadUintN(b io.ByteReader, length uint8) (uint64, error) {
	var res uint64
	for i := uint8(0); i < length; i++ {
		bt, err := b.ReadByte()
		if err != nil {
			return 0, err
		}
		res ^= uint64(bt) << ((length - 1 - i) * 8)
	}
	return res, nil
}

// ReadUint32 reads a uint32
func (bigEndian) ReadUint32(b io.ByteReader) (uint32, error) {
	var b1, b2, b3, b4 uint8
	var err error
	if b4, err = b.ReadByte(); err != nil {
		return 0, err
	}
	if b3, err = b.ReadByte(); err != nil {
		return 0, err
	}
	if b2, err = b.ReadByte(); err != nil {
		return 0, err
	}
	if b1, err = b.ReadByte(); err != nil {
		return 0, err
	}
	return uint32(b1) + uint32(b2)<<8 + uint32(b3)<<16 + uint32(b4)<<24, nil
}

// ReadUint24 reads a uint24
func (bigEndian) ReadUint24(b io.ByteReader) (uint32, error) {
	var b1, b2, b3 uint8
	var err error
	if b3, err = b.ReadByte(); err != nil {
		return 0, err
	}
	if b2, err = b.ReadByte(); err != nil {
		return 0, err
	}
	if b1, err = b.ReadByte(); err != nil {
		return 0, err
	}
	return uint32(b1) + uint32(b2)<<8 + uint32(b3)<<16, nil
}

// ReadUint16 reads a uint16
func (bigEndian) ReadUint16(b io.ByteReader) (uint16, error) {
	var b1, b2 uint8
	var err error
	if b2, err = b.ReadByte(); err != nil {
		return 0, err
	}
	if b1, err = b.ReadByte(); err != nil {
		return 0, err
	}
	return uint16(b1) + uint16(b2)<<8, nil
}

// WriteUint32 writes a uint32
func (bigEndian) WriteUint32(b *bytes.Buffer, i uint32) {
	b.Write([]byte{uint8(i >> 24), uint8(i >> 16), uint8(i >> 8), uint8(i)})
}

// WriteUint24 writes a uint24
func (bigEndian) WriteUint24(b *bytes.Buffer, i uint32) {
	b.Write([]byte{uint8(i >> 16), uint8(i >> 8), uint8(i)})
}

// WriteUint16 writes a uint16
func (bigEndian) WriteUint16(b *bytes.Buffer, i uint16) {
	b.Write([]byte{uint8(i >> 8), uint8(i)})
}
