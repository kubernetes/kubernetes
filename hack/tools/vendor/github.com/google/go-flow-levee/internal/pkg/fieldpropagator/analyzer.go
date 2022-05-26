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

// Package fieldpropagator implements identification of field propagators.
// A field propagator is a function that returns a value tainted by a source field.
package fieldpropagator

import (
	"go/types"
	"reflect"

	"github.com/google/go-flow-levee/internal/pkg/config"
	"github.com/google/go-flow-levee/internal/pkg/fieldtags"
	"github.com/google/go-flow-levee/internal/pkg/propagation"
	"github.com/google/go-flow-levee/internal/pkg/utils"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"
	"golang.org/x/tools/go/ssa"
)

// ResultType is a set of objects that are field propagators.
type ResultType map[types.Object]bool

// IsFieldPropagator determines whether a call is a field propagator.
func (r ResultType) IsFieldPropagator(c *ssa.Call) bool {
	cf, ok := c.Call.Value.(*ssa.Function)
	return ok && r[cf.Object()]
}

type isFieldPropagator struct{}

func (i isFieldPropagator) AFact() {}

func (i isFieldPropagator) String() string {
	return "field propagator identified"
}

var Analyzer = &analysis.Analyzer{
	Name: "fieldpropagator",
	Doc: `This analyzer identifies field propagators.

A field propagator is a function that returns a value that is tainted by a source field.`,
	Flags:      config.FlagSet,
	Run:        run,
	Requires:   []*analysis.Analyzer{buildssa.Analyzer, fieldtags.Analyzer},
	ResultType: reflect.TypeOf(new(ResultType)).Elem(),
	FactTypes:  []analysis.Fact{new(isFieldPropagator)},
}

func run(pass *analysis.Pass) (interface{}, error) {
	taggedFields := pass.ResultOf[fieldtags.Analyzer].(fieldtags.ResultType)
	ssaInput := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA)

	conf, err := config.ReadConfig()
	if err != nil {
		return nil, err
	}

	ssaProg := ssaInput.Pkg.Prog
	for _, mem := range ssaInput.Pkg.Members {
		ssaType, ok := mem.(*ssa.Type)
		if !ok || !conf.IsSourceType(utils.DecomposeType(ssaType.Type())) {
			continue
		}
		for _, meth := range methods(ssaProg, ssaType.Type()) {
			analyzeBlocks(pass, conf, taggedFields, meth)
		}
	}

	isFieldPropagator := map[types.Object]bool{}
	for _, f := range pass.AllObjectFacts() {
		isFieldPropagator[f.Object] = true
	}
	return ResultType(isFieldPropagator), nil
}

func methods(ssaProg *ssa.Program, t types.Type) []*ssa.Function {
	var methods []*ssa.Function
	// The method sets of T and *T are disjoint.
	// In Go code, a variable of type T has access to all
	// the methods of type *T due to an implicit address operation.
	methods = append(methods, methodValues(ssaProg, t)...)
	return append(methods, methodValues(ssaProg, types.NewPointer(t))...)
}

func methodValues(ssaProg *ssa.Program, t types.Type) []*ssa.Function {
	var methodValues []*ssa.Function
	mset := ssaProg.MethodSets.MethodSet(t)
	for i := 0; i < mset.Len(); i++ {
		if meth := ssaProg.MethodValue(mset.At(i)); meth != nil {
			methodValues = append(methodValues, meth)
		}
	}
	return methodValues
}

func analyzeBlocks(pass *analysis.Pass, conf *config.Config, tf fieldtags.ResultType, meth *ssa.Function) {
	var propagations []propagation.Propagation

	for _, b := range meth.Blocks {
		for _, instr := range b.Instrs {
			var (
				txType types.Type
				field  int
			)
			switch t := instr.(type) {
			case *ssa.Field:
				txType = t.X.Type()
				field = t.Field
			case *ssa.FieldAddr:
				txType = t.X.Type()
				field = t.Field
			default:
				continue
			}
			if conf.IsSourceField(utils.DecomposeField(txType, field)) || tf.IsSourceField(txType, field) {
				propagations = append(propagations, propagation.Taint(instr.(ssa.Node), conf, tf))
			}
		}
	}

	for _, b := range meth.Blocks {
		for _, instr := range b.Instrs {
			ret, ok := instr.(*ssa.Return)
			if !ok {
				continue
			}
			for _, prop := range propagations {
				if prop.IsTainted(ret) {
					pass.ExportObjectFact(meth.Object(), &isFieldPropagator{})
				}
			}
		}
	}
}
