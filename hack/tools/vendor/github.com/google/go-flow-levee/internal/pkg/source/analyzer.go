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

package source

import (
	"go/token"
	"reflect"

	"github.com/google/go-flow-levee/internal/pkg/config"
	"github.com/google/go-flow-levee/internal/pkg/fieldpropagator"
	"github.com/google/go-flow-levee/internal/pkg/fieldtags"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"
	"golang.org/x/tools/go/ssa"
)

type ResultType = map[*ssa.Function][]*Source

var Analyzer = &analysis.Analyzer{
	Name:       "source",
	Doc:        "This analyzer identifies ssa.Values that are sources.",
	Flags:      config.FlagSet,
	Run:        run,
	Requires:   []*analysis.Analyzer{buildssa.Analyzer, fieldtags.Analyzer, fieldpropagator.Analyzer},
	ResultType: reflect.TypeOf(new(ResultType)).Elem(),
}

func run(pass *analysis.Pass) (interface{}, error) {
	ssaInput := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA)
	taggedFields := pass.ResultOf[fieldtags.Analyzer].(fieldtags.ResultType)
	fieldPropagators := pass.ResultOf[fieldpropagator.Analyzer].(fieldpropagator.ResultType)

	conf, err := config.ReadConfig()
	if err != nil {
		return nil, err
	}

	sourceMap := identify(conf, ssaInput, taggedFields, fieldPropagators)

	for _, srcs := range sourceMap {
		for _, s := range srcs {
			report(pass, s.Pos())
		}
	}

	return sourceMap, nil
}

func report(pass *analysis.Pass, pos token.Pos) {
	pass.Reportf(pos, "source identified at %s", pass.Fset.Position(pos))
}
