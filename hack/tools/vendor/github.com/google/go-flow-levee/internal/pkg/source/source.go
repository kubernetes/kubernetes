// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package source can be used to identify SSA values that are Sources.
package source

import (
	"go/token"
	"go/types"

	"github.com/google/go-flow-levee/internal/pkg/config"
	"github.com/google/go-flow-levee/internal/pkg/fieldpropagator"
	"github.com/google/go-flow-levee/internal/pkg/fieldtags"
	"github.com/google/go-flow-levee/internal/pkg/sourcetype"
	"github.com/google/go-flow-levee/internal/pkg/utils"
	"golang.org/x/tools/go/analysis/passes/buildssa"
	"golang.org/x/tools/go/ssa"
)

// A Source is a node in the SSA graph that is used as a
// starting point in a propagation analysis.
type Source struct {
	Node ssa.Node
}

// Pos returns the token position of the SSA Node associated with the Source.
func (s *Source) Pos() token.Pos {
	// Extracts don't have a registered position in the source code,
	// so we need to use the position of their related Tuple.
	if e, ok := s.Node.(*ssa.Extract); ok {
		return e.Tuple.Pos()
	}
	// Fields don't *always* have a registered position in the source code,
	// e.g. when accessing an embedded field.
	if f, ok := s.Node.(*ssa.Field); ok && f.Pos() == token.NoPos {
		return f.X.Pos()
	}
	// FieldAddrs don't *always* have a registered position in the source code,
	// e.g. when accessing an embedded field.
	if f, ok := s.Node.(*ssa.FieldAddr); ok && f.Pos() == token.NoPos {
		return f.X.Pos()
	}
	return s.Node.Pos()
}

// New constructs a new Source.
func New(in ssa.Node) *Source {
	return &Source{
		Node: in,
	}
}

// identify individually examines each Function in the SSA code looking for Sources.
// It produces a map relating a Function to the Sources it contains.
// If a Function contains no Sources, it does not appear in the map.
func identify(conf *config.Config, ssaInput *buildssa.SSA, taggedFields fieldtags.ResultType, propagators fieldpropagator.ResultType) map[*ssa.Function][]*Source {
	sourceMap := make(map[*ssa.Function][]*Source)

	for _, fn := range ssaInput.SrcFuncs {
		// no need to analyze the body of sinks, nor of excluded functions
		path, recv, name := utils.DecomposeFunction(fn)
		if conf.IsSink(path, recv, name) || conf.IsExcluded(path, recv, name) {
			continue
		}

		var sources []*Source
		sources = append(sources, sourcesFromParams(fn, conf, taggedFields)...)
		sources = append(sources, sourcesFromClosures(fn, conf, taggedFields)...)
		sources = append(sources, sourcesFromBlocks(fn, conf, taggedFields, propagators)...)

		if len(sources) > 0 {
			sourceMap[fn] = sources
		}
	}
	return sourceMap
}

// sourcesFromParams identifies Sources that appear within a Function's parameters.
func sourcesFromParams(fn *ssa.Function, conf *config.Config, taggedFields fieldtags.ResultType) []*Source {
	var sources []*Source
	for _, p := range fn.Params {
		if sourcetype.IsSourceType(conf, taggedFields, p.Type()) {
			sources = append(sources, New(p))
		}
	}
	return sources
}

// sourcesFromClosures identifies Source values which are captured by closures.
// A value that is captured by a closure will appear as a Free Variable in the
// closure. In the SSA, a Free Variable is represented as a Pointer, distinct
// from the original value.
func sourcesFromClosures(fn *ssa.Function, conf *config.Config, taggedFields fieldtags.ResultType) []*Source {
	var sources []*Source
	for _, fv := range fn.FreeVars {
		if ptr, ok := fv.Type().(*types.Pointer); ok && sourcetype.IsSourceType(conf, taggedFields, ptr) {
			sources = append(sources, New(fv))
		}
	}
	return sources
}

// sourcesFromBlocks finds Source values created by instructions within a function's body.
func sourcesFromBlocks(fn *ssa.Function, conf *config.Config, taggedFields fieldtags.ResultType, propagators fieldpropagator.ResultType) []*Source {
	var sources []*Source
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			if n := instr.(ssa.Node); isSourceNode(n, conf, propagators, taggedFields) {
				sources = append(sources, New(n))
			}
		}
	}
	return sources
}

func isSourceNode(n ssa.Node, conf *config.Config, propagators fieldpropagator.ResultType, taggedFields fieldtags.ResultType) bool {
	switch v := n.(type) {
	// All sources are explicitly identified.
	default:
		return false

	// Values produced by sanitizers are not sources.
	case *ssa.Alloc:
		return !isProducedBySanitizer(v, conf) && sourcetype.IsSourceType(conf, taggedFields, n.(ssa.Value).Type())

	// Values produced by sanitizers are not sources.
	// Values produced by field propagators are.
	case *ssa.Call:
		return !isProducedBySanitizer(v, conf) &&
			(propagators.IsFieldPropagator(v) || sourcetype.IsSourceType(conf, taggedFields, n.(ssa.Value).Type()))

	// A type assertion can assert that an interface is of a source type.
	// Only panicky type asserts will refer to the source Value.
	// The typed value returned in (value, ok) type assertions are examined in the case for ssa.Extract instructions.
	case *ssa.TypeAssert:
		return !v.CommaOk && sourcetype.IsSourceType(conf, taggedFields, v.AssertedType)

	// An Extract is used to obtain a value from an instruction that returns multiple values.
	// In some cases, an extracted value isn't tied to any other instruction that could be used
	// to detect a source value, so the Extract itself has to be used. Specifically:
	// - If the extracted value is a Pointer to a Source
	// - If the extracted value is inlined into a call
	case *ssa.Extract:
		t := v.Tuple.Type().(*types.Tuple).At(v.Index).Type()
		return sourcetype.IsSourceType(conf, taggedFields, t)

	// Unary operator <- can receive sources from a channel.
	case *ssa.UnOp:
		return v.Op == token.ARROW && sourcetype.IsSourceType(conf, taggedFields, n.(ssa.Value).Type())

	// Field access (Field, FieldAddr),
	// collection access (Index, IndexAddr, Lookup),
	// and mutable collection instantiation (MakeMap, MakeChan)
	// are considered sources if they are of source type.
	case *ssa.Field, *ssa.FieldAddr,
		*ssa.Index, *ssa.IndexAddr, *ssa.Lookup,
		*ssa.MakeMap, *ssa.MakeChan:
		return sourcetype.IsSourceType(conf, taggedFields, n.(ssa.Value).Type())
	}
}

func isProducedBySanitizer(v ssa.Value, conf *config.Config) bool {
	for _, instr := range *v.Referrers() {
		store, ok := instr.(*ssa.Store)
		if !ok {
			continue
		}
		call, ok := store.Val.(*ssa.Call)
		if !ok {
			continue
		}
		if callee := call.Call.StaticCallee(); callee != nil && conf.IsSanitizer(utils.DecomposeFunction(callee)) {
			return true
		}
	}
	return false
}
