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

package dwarf

import "strconv"

// DWARF debug info is split into a sequence of compilation units.
// Each unit has its own abbreviation table and address size.

type unit struct {
	base   Offset // byte offset of header within the aggregate info
	off    Offset // byte offset of data within the aggregate info
	data   []byte
	atable abbrevTable
	asize  int
	vers   int
	is64   bool // True for 64-bit DWARF format
}

// Implement the dataFormat interface.

func (u *unit) version() int {
	return u.vers
}

func (u *unit) dwarf64() (bool, bool) {
	return u.is64, true
}

func (u *unit) addrsize() int {
	return u.asize
}

func (d *Data) parseUnits() ([]unit, error) {
	// Count units.
	nunit := 0
	b := makeBuf(d, unknownFormat{}, "info", 0, d.info)
	for len(b.data) > 0 {
		len := b.uint32()
		if len == 0xffffffff {
			len64 := b.uint64()
			if len64 != uint64(uint32(len64)) {
				b.error("unit length overflow")
				break
			}
			len = uint32(len64)
		}
		b.skip(int(len))
		nunit++
	}
	if b.err != nil {
		return nil, b.err
	}

	// Again, this time writing them down.
	b = makeBuf(d, unknownFormat{}, "info", 0, d.info)
	units := make([]unit, nunit)
	for i := range units {
		u := &units[i]
		u.base = b.off
		n := b.uint32()
		if n == 0xffffffff {
			u.is64 = true
			n = uint32(b.uint64())
		}
		vers := b.uint16()
		if vers != 2 && vers != 3 && vers != 4 {
			b.error("unsupported DWARF version " + strconv.Itoa(int(vers)))
			break
		}
		u.vers = int(vers)
		atable, err := d.parseAbbrev(b.uint32())
		if err != nil {
			if b.err == nil {
				b.err = err
			}
			break
		}
		u.atable = atable
		u.asize = int(b.uint8())
		u.off = b.off
		u.data = b.bytes(int(n - (2 + 4 + 1)))
	}
	if b.err != nil {
		return nil, b.err
	}
	return units, nil
}
