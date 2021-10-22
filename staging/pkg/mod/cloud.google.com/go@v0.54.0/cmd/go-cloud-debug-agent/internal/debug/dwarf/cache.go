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

import (
	"fmt"
	"sort"
)

// pcToFuncEntries maps PC ranges to function entries.
//
// Each element contains a *Entry for a function and its corresponding start PC.
// If we know the address one past the last instruction of a function, and it is
// not equal to the start address of the next function, we mark that with
// another element containing that address and a nil entry.  The elements are
// sorted by PC.  Among elements with the same PC, those with non-nil *Entry
// are put earlier.
type pcToFuncEntries []pcToFuncEntry
type pcToFuncEntry struct {
	pc    uint64
	entry *Entry
}

func (p pcToFuncEntries) Len() int      { return len(p) }
func (p pcToFuncEntries) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p pcToFuncEntries) Less(i, j int) bool {
	if p[i].pc != p[j].pc {
		return p[i].pc < p[j].pc
	}
	return p[i].entry != nil && p[j].entry == nil
}

// nameCache maps each symbol name to a linked list of the entries with that name.
type nameCache map[string]*nameCacheEntry
type nameCacheEntry struct {
	entry *Entry
	link  *nameCacheEntry
}

// pcToLineEntries maps PCs to line numbers.
//
// It is a slice of (PC, line, file number) triples, sorted by PC.  The file
// number is an index into the source files slice.
// If (PC1, line1, file1) and (PC2, line2, file2) are two consecutive elements,
// then the span of addresses [PC1, PC2) belongs to (line1, file1).  If an
// element's file number is zero, it only marks the end of a span.
//
// TODO: could save memory by changing pcToLineEntries and lineToPCEntries to use
// interval trees containing references into .debug_line.
type pcToLineEntries []pcToLineEntry
type pcToLineEntry struct {
	pc   uint64
	line uint64
	file uint64
}

func (p pcToLineEntries) Len() int      { return len(p) }
func (p pcToLineEntries) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p pcToLineEntries) Less(i, j int) bool {
	if p[i].pc != p[j].pc {
		return p[i].pc < p[j].pc
	}
	return p[i].file > p[j].file
}

// byFileLine is used temporarily while building lineToPCEntries.
type byFileLine []pcToLineEntry

func (b byFileLine) Len() int      { return len(b) }
func (b byFileLine) Swap(i, j int) { b[i], b[j] = b[j], b[i] }
func (b byFileLine) Less(i, j int) bool {
	if b[i].file != b[j].file {
		return b[i].file < b[j].file
	}
	return b[i].line < b[j].line
}

// lineToPCEntries maps line numbers to breakpoint addresses.
//
// The slice contains, for each source file in Data, a slice of (line, PC)
// pairs, sorted by line.  Note that there may be more than one PC for a line.
type lineToPCEntries [][]lineToPCEntry
type lineToPCEntry struct {
	line uint64
	pc   uint64
}

func (d *Data) buildLineToPCCache(pclfs pcToLineEntries) {
	// TODO: only include lines where is_stmt is true
	sort.Sort(byFileLine(pclfs))
	// Make a slice of (line, PC) pairs for each (non-zero) file.
	var (
		c        = make(lineToPCEntries, len(d.sourceFiles))
		curSlice []lineToPCEntry
	)
	for i, pclf := range pclfs {
		if pclf.file == 0 {
			// This entry indicated the end of an instruction sequence, not a breakpoint.
			continue
		}
		curSlice = append(curSlice, lineToPCEntry{line: pclf.line, pc: pclf.pc})
		if i+1 == len(pclfs) || pclf.file != pclfs[i+1].file {
			// curSlice now contains all of the entries for pclf.file.
			if pclf.file > 0 && pclf.file < uint64(len(c)) {
				c[pclf.file] = curSlice
			}
			curSlice = nil
		}
	}
	d.lineToPCEntries = c
}

func (d *Data) buildPCToLineCache(cache pcToLineEntries) {
	// Sort cache by PC (in increasing order), then by file number (in decreasing order).
	sort.Sort(cache)

	// Build a copy without redundant entries.
	var out pcToLineEntries
	for i, pclf := range cache {
		if i > 0 && pclf.pc == cache[i-1].pc {
			// This entry is for the same PC as the previous entry.
			continue
		}
		if i > 0 && pclf.file == cache[i-1].file && pclf.line == cache[i-1].line {
			// This entry is for the same file and line as the previous entry.
			continue
		}
		out = append(out, pclf)
	}
	d.pcToLineEntries = out
}

// buildLineCaches constructs d.sourceFiles, d.lineToPCEntries, d.pcToLineEntries.
func (d *Data) buildLineCaches() error {
	var cache pcToLineEntries
	r := d.Reader()
	for {
		entry, err := r.Next()
		if err != nil {
			return err
		}
		if entry == nil {
			break
		}
		if entry.Tag != TagCompileUnit {
			r.SkipChildren()
			continue
		}
		// Get the offset of this unit's line number program in .debug_line.
		offset, ok := entry.Val(AttrStmtList).(int64)
		if !ok {
			return fmt.Errorf("AttrStmtList not present or not int64 for unit %d", r.unit)
		}
		buf := makeBuf(d, &d.unit[r.unit], "line", Offset(offset), d.line[offset:])
		var m lineMachine
		if err := m.parseHeader(&buf); err != nil {
			return err
		}
		fileNumOffset := uint64(len(d.sourceFiles))
		for _, f := range m.header.file {
			d.sourceFiles = append(d.sourceFiles, f.name)
		}
		fn := func(m *lineMachine) bool {
			if m.endSequence {
				cache = append(cache, pcToLineEntry{
					pc:   m.address,
					line: 0,
					file: 0,
				})
			} else {
				// m.file is a 1-based index into the files of this line number program's
				// header. Translate it to a 0-based index into d.sourceFiles.
				fnum := fileNumOffset + m.file - 1
				cache = append(cache, pcToLineEntry{
					pc:   m.address,
					line: m.line,
					file: fnum,
				})
			}
			return true
		}
		if err := m.evalCompilationUnit(&buf, fn); err != nil {
			return err
		}
	}
	d.buildLineToPCCache(cache)
	d.buildPCToLineCache(cache)
	return nil
}

// buildInfoCaches initializes nameCache and pcToFuncEntries by walking the
// top-level entries under each compile unit. It swallows any errors in parsing.
func (d *Data) buildInfoCaches() {
	// TODO: record errors somewhere?
	d.nameCache = make(map[string]*nameCacheEntry)

	var pcToFuncEntries pcToFuncEntries

	r := d.Reader()
loop:
	for {
		entry, err := r.Next()
		if entry == nil || err != nil {
			break loop
		}
		if entry.Tag != TagCompileUnit /* DW_TAG_compile_unit */ {
			r.SkipChildren()
			continue
		}
		for {
			entry, err := r.Next()
			if entry == nil || err != nil {
				break loop
			}
			if entry.Tag == 0 {
				// End of children of current compile unit.
				break
			}
			r.SkipChildren()
			// Update name-to-entry cache.
			if name, ok := entry.Val(AttrName).(string); ok {
				d.nameCache[name] = &nameCacheEntry{entry: entry, link: d.nameCache[name]}
			}

			// If this entry is a function, update PC-to-containing-function cache.
			if entry.Tag != TagSubprogram /* DW_TAG_subprogram */ {
				continue
			}

			// DW_AT_low_pc, if present, is the address of the first instruction of
			// the function.
			lowpc, ok := entry.Val(AttrLowpc).(uint64)
			if !ok {
				continue
			}
			pcToFuncEntries = append(pcToFuncEntries, pcToFuncEntry{lowpc, entry})

			// DW_AT_high_pc, if present (TODO: and of class address) is the address
			// one past the last instruction of the function.
			highpc, ok := entry.Val(AttrHighpc).(uint64)
			if !ok {
				continue
			}
			pcToFuncEntries = append(pcToFuncEntries, pcToFuncEntry{highpc, nil})
		}
	}
	// Sort elements by PC.  If there are multiple elements with the same PC,
	// those with non-nil *Entry are placed earlier.
	sort.Sort(pcToFuncEntries)

	// Copy only the first element for each PC to out.
	n := 0
	for i, ce := range pcToFuncEntries {
		if i == 0 || ce.pc != pcToFuncEntries[i-1].pc {
			n++
		}
	}
	out := make([]pcToFuncEntry, 0, n)
	for i, ce := range pcToFuncEntries {
		if i == 0 || ce.pc != pcToFuncEntries[i-1].pc {
			out = append(out, ce)
		}
	}
	d.pcToFuncEntries = out
}
