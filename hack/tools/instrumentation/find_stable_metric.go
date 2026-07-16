/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"fmt"
	"go/ast"

	"k8s.io/component-base/metrics"
)

var metricsOptionStructuresNames = []string{
	"KubeOpts",
	"CounterOpts",
	"GaugeOpts",
	"HistogramOpts",
	"SummaryOpts",
	"TimingHistogramOpts",
}

func findStableMetricDeclaration(tree ast.Node, metricsImportName string) ([]*ast.CallExpr, []error) {
	v := stableMetricFinder{
		metricsImportName:          metricsImportName,
		stableMetricsFunctionCalls: []*ast.CallExpr{},
		errors:                     []error{},
	}
	ast.Walk(&v, tree)
	return v.stableMetricsFunctionCalls, v.errors
}

// Implements visitor pattern for ast.Node that collects all stable metric expressions
type stableMetricFinder struct {
	metricsImportName          string
	currentFunctionCall        *ast.CallExpr
	stableMetricsFunctionCalls []*ast.CallExpr
	errors                     []error
}

var _ ast.Visitor = (*stableMetricFinder)(nil)

func contains(v metrics.StabilityLevel, a []metrics.StabilityLevel) bool {
	for _, i := range a {
		if i == v {
			return true
		}
	}
	return false
}

func (f *stableMetricFinder) Visit(node ast.Node) (w ast.Visitor) {
	switch opts := node.(type) {
	case *ast.CallExpr:
		if se, ok := opts.Fun.(*ast.SelectorExpr); ok {
			if se.Sel.Name == "NewDesc" {
				sl, _ := decodeStabilityLevel(opts.Args[4], f.metricsImportName)
				if sl != nil {
					classes := []metrics.StabilityLevel{metrics.STABLE, metrics.BETA}
					if ALL_STABILITY_CLASSES {
						classes = append(classes, metrics.ALPHA)
					}
					switch {
					case contains(*sl, classes):
						f.stableMetricsFunctionCalls = append(f.stableMetricsFunctionCalls, opts)
						f.currentFunctionCall = nil
					default:
						return nil
					}
				}
			} else {
				f.currentFunctionCall = opts
			}

		} else {
			f.currentFunctionCall = opts
		}
	case *ast.CompositeLit:
		se, ok := opts.Type.(*ast.SelectorExpr)
		if !ok {
			return f
		}
		if !isMetricOps(se.Sel.Name) {
			return f
		}
		id, ok := se.X.(*ast.Ident)
		if !ok {
			return f
		}
		if id.Name != f.metricsImportName {
			return f
		}
		stabilityLevel, err := getStabilityLevel(opts, f.metricsImportName)
		if err != nil {
			f.errors = append(f.errors, err)
			return nil
		}
		classes := []metrics.StabilityLevel{metrics.STABLE, metrics.BETA}
		if ALL_STABILITY_CLASSES {
			classes = append(classes, metrics.ALPHA)
		}
		switch {
		case contains(*stabilityLevel, classes):
			if f.currentFunctionCall == nil {
				f.errors = append(f.errors, newDecodeErrorf(opts, errNotDirectCall))
				return nil
			}
			f.stableMetricsFunctionCalls = append(f.stableMetricsFunctionCalls, f.currentFunctionCall)
			f.currentFunctionCall = nil
		default:
			return nil
		}
	default:
		if f.currentFunctionCall == nil || node == nil || node.Pos() < f.currentFunctionCall.Rparen {
			return f
		}
		f.currentFunctionCall = nil
	}
	return f
}

func isMetricOps(name string) bool {
	var found = false
	for _, optsName := range metricsOptionStructuresNames {
		if name == optsName {
			found = true
			break
		}
	}
	return found
}

func getStabilityLevel(opts *ast.CompositeLit, metricsFrameworkImportName string) (*metrics.StabilityLevel, error) {
	for _, expr := range opts.Elts {
		kv, ok := expr.(*ast.KeyValueExpr)
		if !ok {
			return nil, newDecodeErrorf(expr, errPositionalArguments)
		}
		key := fmt.Sprintf("%v", kv.Key)
		if key != "StabilityLevel" {
			continue
		}
		return decodeStabilityLevel(kv.Value, metricsFrameworkImportName)
	}
	stability := metrics.ALPHA
	return &stability, nil
}
