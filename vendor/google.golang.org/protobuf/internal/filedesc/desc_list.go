// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filedesc

import (
	"fmt"
	"math"
	"sort"
	"sync"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/descfmt"
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/reflect/protoreflect"
	pref "google.golang.org/protobuf/reflect/protoreflect"
)

type FileImports []pref.FileImport

func (p *FileImports) Len() int                            { return len(*p) }
func (p *FileImports) Get(i int) pref.FileImport           { return (*p)[i] }
func (p *FileImports) Format(s fmt.State, r rune)          { descfmt.FormatList(s, r, p) }
func (p *FileImports) ProtoInternal(pragma.DoNotImplement) {}

type Names struct {
	List []pref.Name
	once sync.Once
	has  map[pref.Name]int // protected by once
}

func (p *Names) Len() int                            { return len(p.List) }
func (p *Names) Get(i int) pref.Name                 { return p.List[i] }
func (p *Names) Has(s pref.Name) bool                { return p.lazyInit().has[s] > 0 }
func (p *Names) Format(s fmt.State, r rune)          { descfmt.FormatList(s, r, p) }
func (p *Names) ProtoInternal(pragma.DoNotImplement) {}
func (p *Names) lazyInit() *Names {
	p.once.Do(func() {
		if len(p.List) > 0 {
			p.has = make(map[pref.Name]int, len(p.List))
			for _, s := range p.List {
				p.has[s] = p.has[s] + 1
			}
		}
	})
	return p
}

// CheckValid reports any errors with the set of names with an error message
// that completes the sentence: "ranges is invalid because it has ..."
func (p *Names) CheckValid() error {
	for s, n := range p.lazyInit().has {
		switch {
		case n > 1:
			return errors.New("duplicate name: %q", s)
		case false && !s.IsValid():
			// NOTE: The C++ implementation does not validate the identifier.
			// See https://github.com/protocolbuffers/protobuf/issues/6335.
			return errors.New("invalid name: %q", s)
		}
	}
	return nil
}

type EnumRanges struct {
	List   [][2]pref.EnumNumber // start inclusive; end inclusive
	once   sync.Once
	sorted [][2]pref.EnumNumber // protected by once
}

func (p *EnumRanges) Len() int                     { return len(p.List) }
func (p *EnumRanges) Get(i int) [2]pref.EnumNumber { return p.List[i] }
func (p *EnumRanges) Has(n pref.EnumNumber) bool {
	for ls := p.lazyInit().sorted; len(ls) > 0; {
		i := len(ls) / 2
		switch r := enumRange(ls[i]); {
		case n < r.Start():
			ls = ls[:i] // search lower
		case n > r.End():
			ls = ls[i+1:] // search upper
		default:
			return true
		}
	}
	return false
}
func (p *EnumRanges) Format(s fmt.State, r rune)          { descfmt.FormatList(s, r, p) }
func (p *EnumRanges) ProtoInternal(pragma.DoNotImplement) {}
func (p *EnumRanges) lazyInit() *EnumRanges {
	p.once.Do(func() {
		p.sorted = append(p.sorted, p.List...)
		sort.Slice(p.sorted, func(i, j int) bool {
			return p.sorted[i][0] < p.sorted[j][0]
		})
	})
	return p
}

// CheckValid reports any errors with the set of names with an error message
// that completes the sentence: "ranges is invalid because it has ..."
func (p *EnumRanges) CheckValid() error {
	var rp enumRange
	for i, r := range p.lazyInit().sorted {
		r := enumRange(r)
		switch {
		case !(r.Start() <= r.End()):
			return errors.New("invalid range: %v", r)
		case !(rp.End() < r.Start()) && i > 0:
			return errors.New("overlapping ranges: %v with %v", rp, r)
		}
		rp = r
	}
	return nil
}

type enumRange [2]protoreflect.EnumNumber

func (r enumRange) Start() protoreflect.EnumNumber { return r[0] } // inclusive
func (r enumRange) End() protoreflect.EnumNumber   { return r[1] } // inclusive
func (r enumRange) String() string {
	if r.Start() == r.End() {
		return fmt.Sprintf("%d", r.Start())
	}
	return fmt.Sprintf("%d to %d", r.Start(), r.End())
}

type FieldRanges struct {
	List   [][2]pref.FieldNumber // start inclusive; end exclusive
	once   sync.Once
	sorted [][2]pref.FieldNumber // protected by once
}

func (p *FieldRanges) Len() int                      { return len(p.List) }
func (p *FieldRanges) Get(i int) [2]pref.FieldNumber { return p.List[i] }
func (p *FieldRanges) Has(n pref.FieldNumber) bool {
	for ls := p.lazyInit().sorted; len(ls) > 0; {
		i := len(ls) / 2
		switch r := fieldRange(ls[i]); {
		case n < r.Start():
			ls = ls[:i] // search lower
		case n > r.End():
			ls = ls[i+1:] // search upper
		default:
			return true
		}
	}
	return false
}
func (p *FieldRanges) Format(s fmt.State, r rune)          { descfmt.FormatList(s, r, p) }
func (p *FieldRanges) ProtoInternal(pragma.DoNotImplement) {}
func (p *FieldRanges) lazyInit() *FieldRanges {
	p.once.Do(func() {
		p.sorted = append(p.sorted, p.List...)
		sort.Slice(p.sorted, func(i, j int) bool {
			return p.sorted[i][0] < p.sorted[j][0]
		})
	})
	return p
}

// CheckValid reports any errors with the set of ranges with an error message
// that completes the sentence: "ranges is invalid because it has ..."
func (p *FieldRanges) CheckValid(isMessageSet bool) error {
	var rp fieldRange
	for i, r := range p.lazyInit().sorted {
		r := fieldRange(r)
		switch {
		case !isValidFieldNumber(r.Start(), isMessageSet):
			return errors.New("invalid field number: %d", r.Start())
		case !isValidFieldNumber(r.End(), isMessageSet):
			return errors.New("invalid field number: %d", r.End())
		case !(r.Start() <= r.End()):
			return errors.New("invalid range: %v", r)
		case !(rp.End() < r.Start()) && i > 0:
			return errors.New("overlapping ranges: %v with %v", rp, r)
		}
		rp = r
	}
	return nil
}

// isValidFieldNumber reports whether the field number is valid.
// Unlike the FieldNumber.IsValid method, it allows ranges that cover the
// reserved number range.
func isValidFieldNumber(n protoreflect.FieldNumber, isMessageSet bool) bool {
	if isMessageSet {
		return protowire.MinValidNumber <= n && n <= math.MaxInt32
	}
	return protowire.MinValidNumber <= n && n <= protowire.MaxValidNumber
}

// CheckOverlap reports an error if p and q overlap.
func (p *FieldRanges) CheckOverlap(q *FieldRanges) error {
	rps := p.lazyInit().sorted
	rqs := q.lazyInit().sorted
	for pi, qi := 0, 0; pi < len(rps) && qi < len(rqs); {
		rp := fieldRange(rps[pi])
		rq := fieldRange(rqs[qi])
		if !(rp.End() < rq.Start() || rq.End() < rp.Start()) {
			return errors.New("overlapping ranges: %v with %v", rp, rq)
		}
		if rp.Start() < rq.Start() {
			pi++
		} else {
			qi++
		}
	}
	return nil
}

type fieldRange [2]protoreflect.FieldNumber

func (r fieldRange) Start() protoreflect.FieldNumber { return r[0] }     // inclusive
func (r fieldRange) End() protoreflect.FieldNumber   { return r[1] - 1 } // inclusive
func (r fieldRange) String() string {
	if r.Start() == r.End() {
		return fmt.Sprintf("%d", r.Start())
	}
	return fmt.Sprintf("%d to %d", r.Start(), r.End())
}

type FieldNumbers struct {
	List []pref.FieldNumber
	once sync.Once
	has  map[pref.FieldNumber]struct{} // protected by once
}

func (p *FieldNumbers) Len() int                   { return len(p.List) }
func (p *FieldNumbers) Get(i int) pref.FieldNumber { return p.List[i] }
func (p *FieldNumbers) Has(n pref.FieldNumber) bool {
	p.once.Do(func() {
		if len(p.List) > 0 {
			p.has = make(map[pref.FieldNumber]struct{}, len(p.List))
			for _, n := range p.List {
				p.has[n] = struct{}{}
			}
		}
	})
	_, ok := p.has[n]
	return ok
}
func (p *FieldNumbers) Format(s fmt.State, r rune)          { descfmt.FormatList(s, r, p) }
func (p *FieldNumbers) ProtoInternal(pragma.DoNotImplement) {}

type OneofFields struct {
	List   []pref.FieldDescriptor
	once   sync.Once
	byName map[pref.Name]pref.FieldDescriptor        // protected by once
	byJSON map[string]pref.FieldDescriptor           // protected by once
	byNum  map[pref.FieldNumber]pref.FieldDescriptor // protected by once
}

func (p *OneofFields) Len() int                                         { return len(p.List) }
func (p *OneofFields) Get(i int) pref.FieldDescriptor                   { return p.List[i] }
func (p *OneofFields) ByName(s pref.Name) pref.FieldDescriptor          { return p.lazyInit().byName[s] }
func (p *OneofFields) ByJSONName(s string) pref.FieldDescriptor         { return p.lazyInit().byJSON[s] }
func (p *OneofFields) ByNumber(n pref.FieldNumber) pref.FieldDescriptor { return p.lazyInit().byNum[n] }
func (p *OneofFields) Format(s fmt.State, r rune)                       { descfmt.FormatList(s, r, p) }
func (p *OneofFields) ProtoInternal(pragma.DoNotImplement)              {}

func (p *OneofFields) lazyInit() *OneofFields {
	p.once.Do(func() {
		if len(p.List) > 0 {
			p.byName = make(map[pref.Name]pref.FieldDescriptor, len(p.List))
			p.byJSON = make(map[string]pref.FieldDescriptor, len(p.List))
			p.byNum = make(map[pref.FieldNumber]pref.FieldDescriptor, len(p.List))
			for _, f := range p.List {
				// Field names and numbers are guaranteed to be unique.
				p.byName[f.Name()] = f
				p.byJSON[f.JSONName()] = f
				p.byNum[f.Number()] = f
			}
		}
	})
	return p
}

type SourceLocations struct {
	List []pref.SourceLocation
}

func (p *SourceLocations) Len() int                            { return len(p.List) }
func (p *SourceLocations) Get(i int) pref.SourceLocation       { return p.List[i] }
func (p *SourceLocations) ProtoInternal(pragma.DoNotImplement) {}
