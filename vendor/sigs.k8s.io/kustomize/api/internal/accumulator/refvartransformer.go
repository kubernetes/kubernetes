// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package accumulator

import (
	"sigs.k8s.io/kustomize/api/filters/refvar"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/types"
)

type refVarTransformer struct {
	varMap            map[string]interface{}
	replacementCounts map[string]int
	fieldSpecs        []types.FieldSpec
}

// newRefVarTransformer returns a new refVarTransformer
// that replaces $(VAR) style variables with values.
// The fieldSpecs are the places to look for occurrences of $(VAR).
func newRefVarTransformer(
	varMap map[string]interface{}, fs []types.FieldSpec) *refVarTransformer {
	return &refVarTransformer{
		varMap:     varMap,
		fieldSpecs: fs,
	}
}

// UnusedVars returns slice of Var names that were unused
// after a Transform run.
func (rv *refVarTransformer) UnusedVars() []string {
	var unused []string
	for k := range rv.varMap {
		if _, ok := rv.replacementCounts[k]; !ok {
			unused = append(unused, k)
		}
	}
	return unused
}

// Transform replaces $(VAR) style variables with values.
func (rv *refVarTransformer) Transform(m resmap.ResMap) error {
	rv.replacementCounts = make(map[string]int)
	mf := refvar.MakePrimitiveReplacer(rv.replacementCounts, rv.varMap)
	for _, res := range m.Resources() {
		for _, fieldSpec := range rv.fieldSpecs {
			err := res.ApplyFilter(refvar.Filter{
				MappingFunc: mf,
				FieldSpec:   fieldSpec,
			})
			if err != nil {
				return err
			}
		}
	}
	return nil
}
