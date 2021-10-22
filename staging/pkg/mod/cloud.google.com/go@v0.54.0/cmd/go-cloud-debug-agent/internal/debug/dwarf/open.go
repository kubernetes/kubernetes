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

// Package dwarf provides access to DWARF debugging information loaded from
// executable files, as defined in the DWARF 2.0 Standard at
// http://dwarfstd.org/doc/dwarf-2.0.0.pdf
package dwarf // import "cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/dwarf"

import "encoding/binary"

// Data represents the DWARF debugging information
// loaded from an executable file (for example, an ELF or Mach-O executable).
type Data struct {
	// raw data
	abbrev   []byte
	aranges  []byte
	frame    []byte
	info     []byte
	line     []byte
	pubnames []byte
	ranges   []byte
	str      []byte

	// parsed data
	abbrevCache     map[uint32]abbrevTable
	order           binary.ByteOrder
	typeCache       map[Offset]Type
	typeSigs        map[uint64]*typeUnit
	unit            []unit
	sourceFiles     []string // All the source files listed in .debug_line, from all the compilation units.
	nameCache                // map from name to top-level entries in .debug_info.
	pcToFuncEntries          // cache of .debug_info data for function bounds.
	pcToLineEntries          // cache of .debug_line data, used for efficient PC-to-line mapping.
	lineToPCEntries          // cache of .debug_line data, used for efficient line-to-[]PC mapping.
}

// New returns a new Data object initialized from the given parameters.
// Rather than calling this function directly, clients should typically use
// the DWARF method of the File type of the appropriate package debug/elf,
// debug/macho, or debug/pe.
//
// The []byte arguments are the data from the corresponding debug section
// in the object file; for example, for an ELF object, abbrev is the contents of
// the ".debug_abbrev" section.
func New(abbrev, aranges, frame, info, line, pubnames, ranges, str []byte) (*Data, error) {
	d := &Data{
		abbrev:      abbrev,
		aranges:     aranges,
		frame:       frame,
		info:        info,
		line:        line,
		pubnames:    pubnames,
		ranges:      ranges,
		str:         str,
		abbrevCache: make(map[uint32]abbrevTable),
		typeCache:   make(map[Offset]Type),
		typeSigs:    make(map[uint64]*typeUnit),
	}

	// Sniff .debug_info to figure out byte order.
	// bytes 4:6 are the version, a tiny 16-bit number (1, 2, 3).
	if len(d.info) < 6 {
		return nil, DecodeError{"info", Offset(len(d.info)), "too short"}
	}
	x, y := d.info[4], d.info[5]
	switch {
	case x == 0 && y == 0:
		return nil, DecodeError{"info", 4, "unsupported version 0"}
	case x == 0:
		d.order = binary.BigEndian
	case y == 0:
		d.order = binary.LittleEndian
	default:
		return nil, DecodeError{"info", 4, "cannot determine byte order"}
	}

	u, err := d.parseUnits()
	if err != nil {
		return nil, err
	}
	d.unit = u
	d.buildInfoCaches()
	if err := d.buildLineCaches(); err != nil {
		return nil, err
	}
	return d, nil
}

// AddTypes will add one .debug_types section to the DWARF data.  A
// typical object with DWARF version 4 debug info will have multiple
// .debug_types sections.  The name is used for error reporting only,
// and serves to distinguish one .debug_types section from another.
func (d *Data) AddTypes(name string, types []byte) error {
	return d.parseTypes(name, types)
}
