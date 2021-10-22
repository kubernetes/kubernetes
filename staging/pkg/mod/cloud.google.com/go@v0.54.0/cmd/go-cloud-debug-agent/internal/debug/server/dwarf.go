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

// +build linux

package server

import (
	"errors"
	"fmt"

	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/dwarf"
)

func (s *Server) functionStartAddress(name string) (uint64, error) {
	entry, err := s.dwarfData.LookupFunction(name)
	if err != nil {
		return 0, err
	}
	addrAttr := entry.Val(dwarf.AttrLowpc)
	if addrAttr == nil {
		return 0, fmt.Errorf("symbol %q has no LowPC attribute", name)
	}
	addr, ok := addrAttr.(uint64)
	if !ok {
		return 0, fmt.Errorf("symbol %q has non-uint64 LowPC attribute", name)
	}
	return addr, nil
}

// evalLocation parses a DWARF location description encoded in v.  It works for
// cases where the variable is stored at an offset from the Canonical Frame
// Address.  The return value is this offset.
// TODO: a more general location-description-parsing function.
func evalLocation(v []uint8) (int64, error) {
	// Some DWARF constants.
	const (
		opConsts       = 0x11
		opPlus         = 0x22
		opCallFrameCFA = 0x9C
	)
	if len(v) == 0 {
		return 0, errors.New("empty location specifier")
	}
	if v[0] != opCallFrameCFA {
		return 0, errors.New("unsupported location specifier")
	}
	if len(v) == 1 {
		// The location description was just DW_OP_call_frame_cfa, so the location is exactly the CFA.
		return 0, nil
	}
	if v[1] != opConsts {
		return 0, errors.New("unsupported location specifier")
	}
	offset, v, err := sleb128(v[2:])
	if err != nil {
		return 0, err
	}
	if len(v) == 1 && v[0] == opPlus {
		// The location description was DW_OP_call_frame_cfa, DW_OP_consts <offset>, DW_OP_plus.
		// So return the offset.
		return offset, nil
	}
	return 0, errors.New("unsupported location specifier")
}

func uleb128(v []uint8) (u uint64) {
	var shift uint
	for _, x := range v {
		u |= (uint64(x) & 0x7F) << shift
		shift += 7
		if x&0x80 == 0 {
			break
		}
	}
	return u
}

// sleb128 parses a signed integer encoded with sleb128 at the start of v, and
// returns the integer and the remainder of v.
func sleb128(v []uint8) (s int64, rest []uint8, err error) {
	var shift uint
	var sign int64 = -1
	var i int
	var x uint8
	for i, x = range v {
		s |= (int64(x) & 0x7F) << shift
		shift += 7
		sign <<= 7
		if x&0x80 == 0 {
			if x&0x40 != 0 {
				s |= sign
			}
			break
		}
	}
	if i == len(v) {
		return 0, nil, errors.New("truncated sleb128")
	}
	return s, v[i+1:], nil
}
