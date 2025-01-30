// Copyright 2023 Google LLC
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

// Package ast declares data structures useful for parsed and checked abstract syntax trees
package ast

import (
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// AST contains a protobuf expression and source info along with CEL-native type and reference information.
type AST struct {
	expr       Expr
	sourceInfo *SourceInfo
	typeMap    map[int64]*types.Type
	refMap     map[int64]*ReferenceInfo
}

// Expr returns the root ast.Expr value in the AST.
func (a *AST) Expr() Expr {
	if a == nil {
		return nilExpr
	}
	return a.expr
}

// SourceInfo returns the source metadata associated with the parse / type-check passes.
func (a *AST) SourceInfo() *SourceInfo {
	if a == nil {
		return nil
	}
	return a.sourceInfo
}

// GetType returns the type for the expression at the given id, if one exists, else types.DynType.
func (a *AST) GetType(id int64) *types.Type {
	if t, found := a.TypeMap()[id]; found {
		return t
	}
	return types.DynType
}

// SetType sets the type of the expression node at the given id.
func (a *AST) SetType(id int64, t *types.Type) {
	if a == nil {
		return
	}
	a.typeMap[id] = t
}

// TypeMap returns the map of expression ids to type-checked types.
//
// If the AST is not type-checked, the map will be empty.
func (a *AST) TypeMap() map[int64]*types.Type {
	if a == nil {
		return map[int64]*types.Type{}
	}
	return a.typeMap
}

// GetOverloadIDs returns the set of overload function names for a given expression id.
//
// If the expression id is not a function call, or the AST is not type-checked, the result will be empty.
func (a *AST) GetOverloadIDs(id int64) []string {
	if ref, found := a.ReferenceMap()[id]; found {
		return ref.OverloadIDs
	}
	return []string{}
}

// ReferenceMap returns the map of expression id to identifier, constant, and function references.
func (a *AST) ReferenceMap() map[int64]*ReferenceInfo {
	if a == nil {
		return map[int64]*ReferenceInfo{}
	}
	return a.refMap
}

// SetReference adds a reference to the checked AST type map.
func (a *AST) SetReference(id int64, r *ReferenceInfo) {
	if a == nil {
		return
	}
	a.refMap[id] = r
}

// IsChecked returns whether the AST is type-checked.
func (a *AST) IsChecked() bool {
	return a != nil && len(a.TypeMap()) > 0
}

// NewAST creates a base AST instance with an ast.Expr and ast.SourceInfo value.
func NewAST(e Expr, sourceInfo *SourceInfo) *AST {
	if e == nil {
		e = nilExpr
	}
	return &AST{
		expr:       e,
		sourceInfo: sourceInfo,
		typeMap:    make(map[int64]*types.Type),
		refMap:     make(map[int64]*ReferenceInfo),
	}
}

// NewCheckedAST wraps an parsed AST and augments it with type and reference metadata.
func NewCheckedAST(parsed *AST, typeMap map[int64]*types.Type, refMap map[int64]*ReferenceInfo) *AST {
	return &AST{
		expr:       parsed.Expr(),
		sourceInfo: parsed.SourceInfo(),
		typeMap:    typeMap,
		refMap:     refMap,
	}
}

// Copy creates a deep copy of the Expr and SourceInfo values in the input AST.
//
// Copies of the Expr value are generated using an internal default ExprFactory.
func Copy(a *AST) *AST {
	if a == nil {
		return nil
	}
	e := defaultFactory.CopyExpr(a.expr)
	if !a.IsChecked() {
		return NewAST(e, CopySourceInfo(a.SourceInfo()))
	}
	typesCopy := make(map[int64]*types.Type, len(a.typeMap))
	for id, t := range a.typeMap {
		typesCopy[id] = t
	}
	refsCopy := make(map[int64]*ReferenceInfo, len(a.refMap))
	for id, r := range a.refMap {
		refsCopy[id] = r
	}
	return NewCheckedAST(NewAST(e, CopySourceInfo(a.SourceInfo())), typesCopy, refsCopy)
}

// MaxID returns the upper-bound, non-inclusive, of ids present within the AST's Expr value.
func MaxID(a *AST) int64 {
	visitor := &maxIDVisitor{maxID: 1}
	PostOrderVisit(a.Expr(), visitor)
	for id, call := range a.SourceInfo().MacroCalls() {
		PostOrderVisit(call, visitor)
		if id > visitor.maxID {
			visitor.maxID = id + 1
		}
	}
	return visitor.maxID + 1
}

// NewSourceInfo creates a simple SourceInfo object from an input common.Source value.
func NewSourceInfo(src common.Source) *SourceInfo {
	var lineOffsets []int32
	var desc string
	baseLine := int32(0)
	baseCol := int32(0)
	if src != nil {
		desc = src.Description()
		lineOffsets = src.LineOffsets()
		// Determine whether the source metadata should be computed relative
		// to a base line and column value. This can be determined by requesting
		// the location for offset 0 from the source object.
		if loc, found := src.OffsetLocation(0); found {
			baseLine = int32(loc.Line()) - 1
			baseCol = int32(loc.Column())
		}
	}
	return &SourceInfo{
		desc:         desc,
		lines:        lineOffsets,
		baseLine:     baseLine,
		baseCol:      baseCol,
		offsetRanges: make(map[int64]OffsetRange),
		macroCalls:   make(map[int64]Expr),
	}
}

// CopySourceInfo creates a deep copy of the MacroCalls within the input SourceInfo.
//
// Copies of macro Expr values are generated using an internal default ExprFactory.
func CopySourceInfo(info *SourceInfo) *SourceInfo {
	if info == nil {
		return nil
	}
	rangesCopy := make(map[int64]OffsetRange, len(info.offsetRanges))
	for id, off := range info.offsetRanges {
		rangesCopy[id] = off
	}
	callsCopy := make(map[int64]Expr, len(info.macroCalls))
	for id, call := range info.macroCalls {
		callsCopy[id] = defaultFactory.CopyExpr(call)
	}
	return &SourceInfo{
		syntax:       info.syntax,
		desc:         info.desc,
		lines:        info.lines,
		baseLine:     info.baseLine,
		baseCol:      info.baseCol,
		offsetRanges: rangesCopy,
		macroCalls:   callsCopy,
	}
}

// SourceInfo records basic information about the expression as a textual input and
// as a parsed expression value.
type SourceInfo struct {
	syntax       string
	desc         string
	lines        []int32
	baseLine     int32
	baseCol      int32
	offsetRanges map[int64]OffsetRange
	macroCalls   map[int64]Expr
}

// SyntaxVersion returns the syntax version associated with the text expression.
func (s *SourceInfo) SyntaxVersion() string {
	if s == nil {
		return ""
	}
	return s.syntax
}

// Description provides information about where the expression came from.
func (s *SourceInfo) Description() string {
	if s == nil {
		return ""
	}
	return s.desc
}

// LineOffsets returns a list of the 0-based character offsets in the input text where newlines appear.
func (s *SourceInfo) LineOffsets() []int32 {
	if s == nil {
		return []int32{}
	}
	return s.lines
}

// MacroCalls returns a map of expression id to ast.Expr value where the id represents the expression
// node where the macro was inserted into the AST, and the ast.Expr value represents the original call
// signature which was replaced.
func (s *SourceInfo) MacroCalls() map[int64]Expr {
	if s == nil {
		return map[int64]Expr{}
	}
	return s.macroCalls
}

// GetMacroCall returns the original ast.Expr value for the given expression if it was generated via
// a macro replacement.
//
// Note, parsing options must be enabled to track macro calls before this method will return a value.
func (s *SourceInfo) GetMacroCall(id int64) (Expr, bool) {
	e, found := s.MacroCalls()[id]
	return e, found
}

// SetMacroCall records a macro call at a specific location.
func (s *SourceInfo) SetMacroCall(id int64, e Expr) {
	if s != nil {
		s.macroCalls[id] = e
	}
}

// ClearMacroCall removes the macro call at the given expression id.
func (s *SourceInfo) ClearMacroCall(id int64) {
	if s != nil {
		delete(s.macroCalls, id)
	}
}

// OffsetRanges returns a map of expression id to OffsetRange values where the range indicates either:
// the start and end position in the input stream where the expression occurs, or the start position
// only. If the range only captures start position, the stop position of the range will be equal to
// the start.
func (s *SourceInfo) OffsetRanges() map[int64]OffsetRange {
	if s == nil {
		return map[int64]OffsetRange{}
	}
	return s.offsetRanges
}

// GetOffsetRange retrieves an OffsetRange for the given expression id if one exists.
func (s *SourceInfo) GetOffsetRange(id int64) (OffsetRange, bool) {
	if s == nil {
		return OffsetRange{}, false
	}
	o, found := s.offsetRanges[id]
	return o, found
}

// SetOffsetRange sets the OffsetRange for the given expression id.
func (s *SourceInfo) SetOffsetRange(id int64, o OffsetRange) {
	if s == nil {
		return
	}
	s.offsetRanges[id] = o
}

// ClearOffsetRange removes the OffsetRange for the given expression id.
func (s *SourceInfo) ClearOffsetRange(id int64) {
	if s != nil {
		delete(s.offsetRanges, id)
	}
}

// GetStartLocation calculates the human-readable 1-based line and 0-based column of the first character
// of the expression node at the id.
func (s *SourceInfo) GetStartLocation(id int64) common.Location {
	if o, found := s.GetOffsetRange(id); found {
		return s.GetLocationByOffset(o.Start)
	}
	return common.NoLocation
}

// GetStopLocation calculates the human-readable 1-based line and 0-based column of the last character for
// the expression node at the given id.
//
// If the SourceInfo was generated from a serialized protobuf representation, the stop location will
// be identical to the start location for the expression.
func (s *SourceInfo) GetStopLocation(id int64) common.Location {
	if o, found := s.GetOffsetRange(id); found {
		return s.GetLocationByOffset(o.Stop)
	}
	return common.NoLocation
}

// GetLocationByOffset returns the line and column information for a given character offset.
func (s *SourceInfo) GetLocationByOffset(offset int32) common.Location {
	line := 1
	col := int(offset)
	for _, lineOffset := range s.LineOffsets() {
		if lineOffset > offset {
			break
		}
		line++
		col = int(offset - lineOffset)
	}
	return common.NewLocation(line, col)
}

// ComputeOffset calculates the 0-based character offset from a 1-based line and 0-based column.
func (s *SourceInfo) ComputeOffset(line, col int32) int32 {
	if s != nil {
		line = s.baseLine + line
		col = s.baseCol + col
	}
	if line == 1 {
		return col
	}
	if line < 1 || line > int32(len(s.LineOffsets())) {
		return -1
	}
	offset := s.LineOffsets()[line-2]
	return offset + col
}

// OffsetRange captures the start and stop positions of a section of text in the input expression.
type OffsetRange struct {
	Start int32
	Stop  int32
}

// ReferenceInfo contains a CEL native representation of an identifier reference which may refer to
// either a qualified identifier name, a set of overload ids, or a constant value from an enum.
type ReferenceInfo struct {
	Name        string
	OverloadIDs []string
	Value       ref.Val
}

// NewIdentReference creates a ReferenceInfo instance for an identifier with an optional constant value.
func NewIdentReference(name string, value ref.Val) *ReferenceInfo {
	return &ReferenceInfo{Name: name, Value: value}
}

// NewFunctionReference creates a ReferenceInfo instance for a set of function overloads.
func NewFunctionReference(overloads ...string) *ReferenceInfo {
	info := &ReferenceInfo{}
	for _, id := range overloads {
		info.AddOverload(id)
	}
	return info
}

// AddOverload appends a function overload ID to the ReferenceInfo.
func (r *ReferenceInfo) AddOverload(overloadID string) {
	for _, id := range r.OverloadIDs {
		if id == overloadID {
			return
		}
	}
	r.OverloadIDs = append(r.OverloadIDs, overloadID)
}

// Equals returns whether two references are identical to each other.
func (r *ReferenceInfo) Equals(other *ReferenceInfo) bool {
	if r.Name != other.Name {
		return false
	}
	if len(r.OverloadIDs) != len(other.OverloadIDs) {
		return false
	}
	if len(r.OverloadIDs) != 0 {
		overloadMap := make(map[string]struct{}, len(r.OverloadIDs))
		for _, id := range r.OverloadIDs {
			overloadMap[id] = struct{}{}
		}
		for _, id := range other.OverloadIDs {
			_, found := overloadMap[id]
			if !found {
				return false
			}
		}
	}
	if r.Value == nil && other.Value == nil {
		return true
	}
	if r.Value == nil && other.Value != nil ||
		r.Value != nil && other.Value == nil ||
		r.Value.Equal(other.Value) != types.True {
		return false
	}
	return true
}

type maxIDVisitor struct {
	maxID int64
	*baseVisitor
}

// VisitExpr updates the max identifier if the incoming expression id is greater than previously observed.
func (v *maxIDVisitor) VisitExpr(e Expr) {
	if v.maxID < e.ID() {
		v.maxID = e.ID()
	}
}

// VisitEntryExpr updates the max identifier if the incoming entry id is greater than previously observed.
func (v *maxIDVisitor) VisitEntryExpr(e EntryExpr) {
	if v.maxID < e.ID() {
		v.maxID = e.ID()
	}
}
