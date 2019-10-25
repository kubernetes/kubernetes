// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package accumulator

import (
	"fmt"

	expansion2 "sigs.k8s.io/kustomize/api/internal/accumulator/expansion"

	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/transform"
	"sigs.k8s.io/kustomize/api/types"
)

type refVarTransformer struct {
	varMap            map[string]interface{}
	replacementCounts map[string]int
	fieldSpecs        []types.FieldSpec
	mappingFunc       func(string) interface{}
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

// replaceVars accepts as 'in' a string, or string array, which can have
// embedded instances of $VAR style variables, e.g. a container command string.
// The function returns the string with the variables expanded to their final
// values.
func (rv *refVarTransformer) replaceVars(in interface{}) (interface{}, error) {
	switch vt := in.(type) {
	case []interface{}:
		var xs []interface{}
		for _, a := range in.([]interface{}) {
			xs = append(xs, expansion2.Expand(a.(string), rv.mappingFunc))
		}
		return xs, nil
	case map[string]interface{}:
		inMap := in.(map[string]interface{})
		xs := make(map[string]interface{}, len(inMap))
		for k, v := range inMap {
			s, ok := v.(string)
			if !ok {
				// This field not contain a $(VAR) since it is not
				// of string type. For instance .spec.replicas: 3 in
				// a Deployment object
				xs[k] = v
			} else {
				// This field can potentially contains a $(VAR) since it is
				// of string type. For instance .spec.replicas: $(REPLICAS)
				// in a Deployment object
				xs[k] = expansion2.Expand(s, rv.mappingFunc)
			}
		}
		return xs, nil
	case interface{}:
		s, ok := in.(string)
		if !ok {
			// This field not contain a $(VAR) since it is not of string type.
			return in, nil
		}
		// This field can potentially contain a $(VAR) since it is
		// of string type.
		return expansion2.Expand(s, rv.mappingFunc), nil
	case nil:
		return nil, nil
	default:
		return "", fmt.Errorf("invalid type encountered %T", vt)
	}
}

// UnusedVars returns slice of Var names that were unused
// after a Transform run.
func (rv *refVarTransformer) UnusedVars() []string {
	var unused []string
	for k := range rv.varMap {
		_, ok := rv.replacementCounts[k]
		if !ok {
			unused = append(unused, k)
		}
	}
	return unused
}

// Transform replaces $(VAR) style variables with values.
func (rv *refVarTransformer) Transform(m resmap.ResMap) error {
	rv.replacementCounts = make(map[string]int)
	rv.mappingFunc = expansion2.MappingFuncFor(
		rv.replacementCounts, rv.varMap)
	for _, res := range m.Resources() {
		for _, fieldSpec := range rv.fieldSpecs {
			if res.OrgId().IsSelected(&fieldSpec.Gvk) {
				if err := transform.MutateField(
					res.Map(), fieldSpec.PathSlice(),
					false, rv.replaceVars); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
