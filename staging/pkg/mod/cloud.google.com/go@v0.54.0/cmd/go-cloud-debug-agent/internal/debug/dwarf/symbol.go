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

// This file provides simple methods to access the symbol table by name and address.

import (
	"fmt"
	"regexp"
	"sort"
)

// lookupEntry returns the first Entry for the name.
// If tag is non-zero, only entries with that tag are considered.
func (d *Data) lookupEntry(name string, tag Tag) (*Entry, error) {
	x, ok := d.nameCache[name]
	if !ok {
		return nil, fmt.Errorf("DWARF entry for %q not found", name)
	}
	for ; x != nil; x = x.link {
		if tag == 0 || x.entry.Tag == tag {
			return x.entry, nil
		}
	}
	return nil, fmt.Errorf("no DWARF entry for %q with tag %s", name, tag)
}

// LookupMatchingSymbols returns the names of all top-level entries matching
// the given regular expression.
func (d *Data) LookupMatchingSymbols(nameRE *regexp.Regexp) (result []string, err error) {
	for name := range d.nameCache {
		if nameRE.MatchString(name) {
			result = append(result, name)
		}
	}
	return result, nil
}

// LookupEntry returns the Entry for the named symbol.
func (d *Data) LookupEntry(name string) (*Entry, error) {
	return d.lookupEntry(name, 0)
}

// LookupFunction returns the entry for a function.
func (d *Data) LookupFunction(name string) (*Entry, error) {
	return d.lookupEntry(name, TagSubprogram)
}

// LookupVariable returns the entry for a (global) variable.
func (d *Data) LookupVariable(name string) (*Entry, error) {
	return d.lookupEntry(name, TagVariable)
}

// EntryLocation returns the address of the object referred to by the given Entry.
func (d *Data) EntryLocation(e *Entry) (uint64, error) {
	loc, _ := e.Val(AttrLocation).([]byte)
	if len(loc) == 0 {
		return 0, fmt.Errorf("DWARF entry has no Location attribute")
	}
	// TODO: implement the DWARF Location bytecode. What we have here only
	// recognizes a program with a single literal opAddr bytecode.
	if asize := d.unit[0].asize; loc[0] == opAddr && len(loc) == 1+asize {
		switch asize {
		case 1:
			return uint64(loc[1]), nil
		case 2:
			return uint64(d.order.Uint16(loc[1:])), nil
		case 4:
			return uint64(d.order.Uint32(loc[1:])), nil
		case 8:
			return d.order.Uint64(loc[1:]), nil
		}
	}
	return 0, fmt.Errorf("DWARF entry has an unimplemented Location op")
}

// EntryType returns the Type for an Entry.
func (d *Data) EntryType(e *Entry) (Type, error) {
	off, err := d.EntryTypeOffset(e)
	if err != nil {
		return nil, err
	}
	return d.Type(off)
}

// EntryTypeOffset returns the offset in the given Entry's type attribute.
func (d *Data) EntryTypeOffset(e *Entry) (Offset, error) {
	v := e.Val(AttrType)
	if v == nil {
		return 0, fmt.Errorf("DWARF entry has no Type attribute")
	}
	off, ok := v.(Offset)
	if !ok {
		return 0, fmt.Errorf("DWARF entry has an invalid Type attribute")
	}
	return off, nil
}

// PCToFunction returns the entry and address for the function containing the
// specified PC.
func (d *Data) PCToFunction(pc uint64) (entry *Entry, lowpc uint64, err error) {
	p := d.pcToFuncEntries
	if len(p) == 0 {
		return nil, 0, fmt.Errorf("no function addresses loaded")
	}
	i := sort.Search(len(p), func(i int) bool { return p[i].pc > pc }) - 1
	// The search failed if:
	// - pc was before the start of any function.
	// - The largest function bound not larger than pc was the end of a function,
	//   not the start of one.
	// - The largest function bound not larger than pc was the start of a function
	//   that we don't know the end of, and the PC is much larger than the start.
	if i == -1 || p[i].entry == nil || (i+1 == len(p) && pc-p[i].pc >= 1<<20) {
		return nil, 0, fmt.Errorf("no function at %x", pc)
	}
	return p[i].entry, p[i].pc, nil
}
