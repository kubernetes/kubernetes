// Copyright 2018 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package binaryutil contains convenience wrappers around encoding/binary.
package binaryutil

import (
	"bytes"
	"encoding/binary"
	"unsafe"
)

// ByteOrder is like binary.ByteOrder, but allocates memory and returns byte
// slices, for convenience.
type ByteOrder interface {
	PutUint16(v uint16) []byte
	PutUint32(v uint32) []byte
	PutUint64(v uint64) []byte
	Uint16(b []byte) uint16
	Uint32(b []byte) uint32
	Uint64(b []byte) uint64
}

// NativeEndian is either little endian or big endian, depending on the native
// endian-ness, and allocates memory and returns byte slices, for convenience.
var NativeEndian ByteOrder = &nativeEndian{}

type nativeEndian struct{}

func (nativeEndian) PutUint16(v uint16) []byte {
	buf := make([]byte, 2)
	*(*uint16)(unsafe.Pointer(&buf[0])) = v
	return buf
}

func (nativeEndian) PutUint32(v uint32) []byte {
	buf := make([]byte, 4)
	*(*uint32)(unsafe.Pointer(&buf[0])) = v
	return buf
}

func (nativeEndian) PutUint64(v uint64) []byte {
	buf := make([]byte, 8)
	*(*uint64)(unsafe.Pointer(&buf[0])) = v
	return buf
}

func (nativeEndian) Uint16(b []byte) uint16 {
	return *(*uint16)(unsafe.Pointer(&b[0]))
}

func (nativeEndian) Uint32(b []byte) uint32 {
	return *(*uint32)(unsafe.Pointer(&b[0]))
}

func (nativeEndian) Uint64(b []byte) uint64 {
	return *(*uint64)(unsafe.Pointer(&b[0]))
}

// BigEndian is like binary.BigEndian, but allocates memory and returns byte
// slices, for convenience.
var BigEndian ByteOrder = &bigEndian{}

type bigEndian struct{}

func (bigEndian) PutUint16(v uint16) []byte {
	buf := make([]byte, 2)
	binary.BigEndian.PutUint16(buf, v)
	return buf
}

func (bigEndian) PutUint32(v uint32) []byte {
	buf := make([]byte, 4)
	binary.BigEndian.PutUint32(buf, v)
	return buf
}

func (bigEndian) PutUint64(v uint64) []byte {
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, v)
	return buf
}

func (bigEndian) Uint16(b []byte) uint16 {
	return binary.BigEndian.Uint16(b)
}

func (bigEndian) Uint32(b []byte) uint32 {
	return binary.BigEndian.Uint32(b)
}

func (bigEndian) Uint64(b []byte) uint64 {
	return binary.BigEndian.Uint64(b)
}

// For dealing with types not supported by the encoding/binary interface

func PutInt32(v int32) []byte {
	buf := make([]byte, 4)
	*(*int32)(unsafe.Pointer(&buf[0])) = v
	return buf
}

func Int32(b []byte) int32 {
	return *(*int32)(unsafe.Pointer(&b[0]))
}

func PutString(s string) []byte {
	return []byte(s)
}

func String(b []byte) string {
	return string(bytes.TrimRight(b, "\x00"))
}
