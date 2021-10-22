// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Functions for reading values of various types from a program's memory.

// +build linux

package server

import (
	"errors"
	"fmt"

	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/dwarf"
)

// peekBytes reads len(buf) bytes at addr.
func (s *Server) peekBytes(addr uint64, buf []byte) error {
	return s.ptracePeek(s.stoppedPid, uintptr(addr), buf)
}

// peekPtr reads a pointer at addr.
func (s *Server) peekPtr(addr uint64) (uint64, error) {
	buf := make([]byte, s.arch.PointerSize)
	if err := s.peekBytes(addr, buf); err != nil {
		return 0, err
	}
	return s.arch.Uintptr(buf), nil
}

// peekUint8 reads a single byte at addr.
func (s *Server) peekUint8(addr uint64) (byte, error) {
	buf := make([]byte, 1)
	if err := s.peekBytes(addr, buf); err != nil {
		return 0, err
	}
	return uint8(s.arch.UintN(buf)), nil
}

// peekInt reads an int of size n bytes at addr.
func (s *Server) peekInt(addr uint64, n int64) (int64, error) {
	buf := make([]byte, n)
	if err := s.peekBytes(addr, buf); err != nil {
		return 0, err
	}
	return s.arch.IntN(buf), nil
}

// peekUint reads a uint of size n bytes at addr.
func (s *Server) peekUint(addr uint64, n int64) (uint64, error) {
	buf := make([]byte, n)
	if err := s.peekBytes(addr, buf); err != nil {
		return 0, err
	}
	return s.arch.UintN(buf), nil
}

// peekSlice reads the header of a slice with the given type and address.
func (s *Server) peekSlice(t *dwarf.SliceType, addr uint64) (debug.Slice, error) {
	ptr, err := s.peekPtrStructField(&t.StructType, addr, "array")
	if err != nil {
		return debug.Slice{}, fmt.Errorf("reading slice location: %s", err)
	}
	length, err := s.peekUintOrIntStructField(&t.StructType, addr, "len")
	if err != nil {
		return debug.Slice{}, fmt.Errorf("reading slice length: %s", err)
	}
	capacity, err := s.peekUintOrIntStructField(&t.StructType, addr, "cap")
	if err != nil {
		return debug.Slice{}, fmt.Errorf("reading slice capacity: %s", err)
	}
	if capacity < length {
		return debug.Slice{}, fmt.Errorf("slice's capacity %d is less than its length %d", capacity, length)
	}

	return debug.Slice{
		debug.Array{
			ElementTypeID: uint64(t.ElemType.Common().Offset),
			Address:       uint64(ptr),
			Length:        length,
			StrideBits:    uint64(t.ElemType.Common().ByteSize) * 8,
		},
		capacity,
	}, nil
}

// peekString reads a string of the given type at the given address.
// At most byteLimit bytes will be read.  If the string is longer, "..." is appended.
func (s *Server) peekString(typ *dwarf.StringType, a uint64, byteLimit uint64) (string, error) {
	ptr, err := s.peekPtrStructField(&typ.StructType, a, "str")
	if err != nil {
		return "", err
	}
	length, err := s.peekUintOrIntStructField(&typ.StructType, a, "len")
	if err != nil {
		return "", err
	}
	if length > byteLimit {
		buf := make([]byte, byteLimit, byteLimit+3)
		if err := s.peekBytes(ptr, buf); err != nil {
			return "", err
		} else {
			buf = append(buf, '.', '.', '.')
			return string(buf), nil
		}
	} else {
		buf := make([]byte, length)
		if err := s.peekBytes(ptr, buf); err != nil {
			return "", err
		} else {
			return string(buf), nil
		}
	}
}

// peekCString reads a NUL-terminated string at the given address.
// At most byteLimit bytes will be read.  If the string is longer, "..." is appended.
// peekCString never returns errors; if an error occurs, the string will be truncated in some way.
func (s *Server) peekCString(a uint64, byteLimit uint64) string {
	buf := make([]byte, byteLimit, byteLimit+3)
	s.peekBytes(a, buf)
	for i, c := range buf {
		if c == 0 {
			return string(buf[0:i])
		}
	}
	buf = append(buf, '.', '.', '.')
	return string(buf)
}

// peekPtrStructField reads a pointer in the field fieldName of the struct
// of type t at addr.
func (s *Server) peekPtrStructField(t *dwarf.StructType, addr uint64, fieldName string) (uint64, error) {
	f, err := getField(t, fieldName)
	if err != nil {
		return 0, fmt.Errorf("reading field %s: %s", fieldName, err)
	}
	if _, ok := f.Type.(*dwarf.PtrType); !ok {
		return 0, fmt.Errorf("field %s is not a pointer", fieldName)
	}
	return s.peekPtr(addr + uint64(f.ByteOffset))
}

// peekUintOrIntStructField reads a signed or unsigned integer in the field fieldName
// of the struct of type t at addr. If the value is negative, it returns an error.
// This function is used when the value should be non-negative, but the DWARF
// type of the field may be signed or unsigned.
func (s *Server) peekUintOrIntStructField(t *dwarf.StructType, addr uint64, fieldName string) (uint64, error) {
	f, err := getField(t, fieldName)
	if err != nil {
		return 0, fmt.Errorf("reading field %s: %s", fieldName, err)
	}
	ut, ok := f.Type.(*dwarf.UintType)
	if ok {
		return s.peekUint(addr+uint64(f.ByteOffset), ut.ByteSize)
	}
	it, ok := f.Type.(*dwarf.IntType)
	if !ok {
		return 0, fmt.Errorf("field %s is not an integer", fieldName)
	}
	i, err := s.peekInt(addr+uint64(f.ByteOffset), it.ByteSize)
	if err != nil {
		return 0, err
	}
	if i < 0 {
		return 0, fmt.Errorf("field %s is negative", fieldName)
	}
	return uint64(i), nil
}

// peekUintStructField reads a uint in the field fieldName of the struct
// of type t at addr.  The size of the uint is determined by the field.
func (s *Server) peekUintStructField(t *dwarf.StructType, addr uint64, fieldName string) (uint64, error) {
	f, err := getField(t, fieldName)
	if err != nil {
		return 0, fmt.Errorf("reading field %s: %s", fieldName, err)
	}
	ut, ok := f.Type.(*dwarf.UintType)
	if !ok {
		return 0, fmt.Errorf("field %s is not an unsigned integer", fieldName)
	}
	return s.peekUint(addr+uint64(f.ByteOffset), ut.ByteSize)
}

// peekIntStructField reads an int in the field fieldName of the struct
// of type t at addr.  The size of the int is determined by the field.
func (s *Server) peekIntStructField(t *dwarf.StructType, addr uint64, fieldName string) (int64, error) {
	f, err := getField(t, fieldName)
	if err != nil {
		return 0, fmt.Errorf("reading field %s: %s", fieldName, err)
	}
	it, ok := f.Type.(*dwarf.IntType)
	if !ok {
		return 0, fmt.Errorf("field %s is not a signed integer", fieldName)
	}
	return s.peekInt(addr+uint64(f.ByteOffset), it.ByteSize)
}

// peekStringStructField reads a string field from the struct of the given type
// at the given address.
// At most byteLimit bytes will be read.  If the string is longer, "..." is appended.
func (s *Server) peekStringStructField(t *dwarf.StructType, addr uint64, fieldName string, byteLimit uint64) (string, error) {
	f, err := getField(t, fieldName)
	if err != nil {
		return "", fmt.Errorf("reading field %s: %s", fieldName, err)
	}
	st, ok := followTypedefs(f.Type).(*dwarf.StringType)
	if !ok {
		return "", fmt.Errorf("field %s is not a string", fieldName)
	}
	return s.peekString(st, addr+uint64(f.ByteOffset), byteLimit)
}

// peekMapLocationAndType returns the address and DWARF type of the underlying
// struct of a map variable.
func (s *Server) peekMapLocationAndType(t *dwarf.MapType, a uint64) (uint64, *dwarf.StructType, error) {
	// Maps are pointers to structs.
	pt, ok := t.Type.(*dwarf.PtrType)
	if !ok {
		return 0, nil, errors.New("bad map type: not a pointer")
	}
	st, ok := pt.Type.(*dwarf.StructType)
	if !ok {
		return 0, nil, errors.New("bad map type: not a pointer to a struct")
	}
	// a is the address of a pointer to a struct.  Get the pointer's value.
	a, err := s.peekPtr(a)
	if err != nil {
		return 0, nil, fmt.Errorf("reading map pointer: %s", err)
	}
	return a, st, nil
}

// peekMapValues reads a map at the given address and calls fn with the addresses for each (key, value) pair.
// If fn returns false, peekMapValues stops.
func (s *Server) peekMapValues(t *dwarf.MapType, a uint64, fn func(keyAddr, valAddr uint64, keyType, valType dwarf.Type) bool) error {
	a, st, err := s.peekMapLocationAndType(t, a)
	if err != nil {
		return err
	}
	if a == 0 {
		// The pointer was nil, so the map is empty.
		return nil
	}
	// Gather information about the struct type and the map bucket type.
	b, err := s.peekUintStructField(st, a, "B")
	if err != nil {
		return fmt.Errorf("reading map: %s", err)
	}
	buckets, err := s.peekPtrStructField(st, a, "buckets")
	if err != nil {
		return fmt.Errorf("reading map: %s", err)
	}
	oldbuckets, err := s.peekPtrStructField(st, a, "oldbuckets")
	if err != nil {
		return fmt.Errorf("reading map: %s", err)
	}
	bf, err := getField(st, "buckets")
	if err != nil {
		return fmt.Errorf("reading map: %s", err)
	}
	bucketPtrType, ok := bf.Type.(*dwarf.PtrType)
	if !ok {
		return errors.New("bad map bucket type: not a pointer")
	}
	bt, ok := bucketPtrType.Type.(*dwarf.StructType)
	if !ok {
		return errors.New("bad map bucket type: not a pointer to a struct")
	}
	bucketSize := uint64(bucketPtrType.Type.Size())
	tophashField, err := getField(bt, "tophash")
	if err != nil {
		return fmt.Errorf("reading map: %s", err)
	}
	bucketCnt := uint64(tophashField.Type.Size())
	tophashFieldOffset := uint64(tophashField.ByteOffset)
	keysField, err := getField(bt, "keys")
	if err != nil {
		return fmt.Errorf("reading map: %s", err)
	}
	keysType, ok := keysField.Type.(*dwarf.ArrayType)
	if !ok {
		return errors.New(`bad map bucket type: "keys" is not an array`)
	}
	keyType := keysType.Type
	keysStride := uint64(keysType.StrideBitSize / 8)
	keysFieldOffset := uint64(keysField.ByteOffset)
	valuesField, err := getField(bt, "values")
	if err != nil {
		return fmt.Errorf("reading map: %s", err)
	}
	valuesType, ok := valuesField.Type.(*dwarf.ArrayType)
	if !ok {
		return errors.New(`bad map bucket type: "values" is not an array`)
	}
	valueType := valuesType.Type
	valuesStride := uint64(valuesType.StrideBitSize / 8)
	valuesFieldOffset := uint64(valuesField.ByteOffset)
	overflowField, err := getField(bt, "overflow")
	if err != nil {
		return fmt.Errorf("reading map: %s", err)
	}
	overflowFieldOffset := uint64(overflowField.ByteOffset)

	// Iterate through the two arrays of buckets.
	bucketArrays := [2]struct {
		addr uint64
		size uint64
	}{
		{buckets, 1 << b},
		{oldbuckets, 1 << (b - 1)},
	}
	for _, bucketArray := range bucketArrays {
		if bucketArray.addr == 0 {
			continue
		}
		for i := uint64(0); i < bucketArray.size; i++ {
			bucketAddr := bucketArray.addr + i*bucketSize
			// Iterate through the linked list of buckets.
			// TODO: check for repeated bucket pointers.
			for bucketAddr != 0 {
				// Iterate through each entry in the bucket.
				for j := uint64(0); j < bucketCnt; j++ {
					tophash, err := s.peekUint8(bucketAddr + tophashFieldOffset + j)
					if err != nil {
						return errors.New("reading map: " + err.Error())
					}
					// From runtime/hashmap.go
					const minTopHash = 4
					if tophash < minTopHash {
						continue
					}
					keyAddr := bucketAddr + keysFieldOffset + j*keysStride
					valAddr := bucketAddr + valuesFieldOffset + j*valuesStride
					if !fn(keyAddr, valAddr, keyType, valueType) {
						return nil
					}
				}
				var err error
				bucketAddr, err = s.peekPtr(bucketAddr + overflowFieldOffset)
				if err != nil {
					return errors.New("reading map: " + err.Error())
				}
			}
		}
	}

	return nil
}

// peekMapLength returns the number of elements in a map at the given address.
func (s *Server) peekMapLength(t *dwarf.MapType, a uint64) (uint64, error) {
	a, st, err := s.peekMapLocationAndType(t, a)
	if err != nil {
		return 0, err
	}
	if a == 0 {
		// The pointer was nil, so the map is empty.
		return 0, nil
	}
	length, err := s.peekUintOrIntStructField(st, a, "count")
	if err != nil {
		return 0, fmt.Errorf("reading map: %s", err)
	}
	return uint64(length), nil
}
