/*
Copyright 2018 The Kubernetes Authors.

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

package transformers

import (
	"fmt"
	"sigs.k8s.io/kustomize/pkg/expansion"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

type RefVarTransformer struct {
	varMap            map[string]interface{}
	replacementCounts map[string]int
	fieldSpecs        []config.FieldSpec
	mappingFunc       func(string) interface{}
}

// NewRefVarTransformer returns a new RefVarTransformer
// that replaces $(VAR) style variables with values.
// The fieldSpecs are the places to look for occurrences of $(VAR).
func NewRefVarTransformer(
	varMap map[string]interface{}, fs []config.FieldSpec) *RefVarTransformer {
	return &RefVarTransformer{
		varMap:     varMap,
		fieldSpecs: fs,
	}
}

// replaceVars accepts as 'in' a string, or string array, which can have
// embedded instances of $VAR style variables, e.g. a container command string.
// The function returns the string with the variables expanded to their final
// values.
func (rv *RefVarTransformer) replaceVars(in interface{}) (interface{}, error) {
	switch vt := in.(type) {
	case []interface{}:
		var xs []interface{}
		for _, a := range in.([]interface{}) {
			xs = append(xs, expansion.Expand(a.(string), rv.mappingFunc))
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
				xs[k] = expansion.Expand(s, rv.mappingFunc)
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
		return expansion.Expand(s, rv.mappingFunc), nil
	case nil:
		return nil, nil
	default:
		return "", fmt.Errorf("invalid type encountered %T", vt)
	}
}

// UnusedVars returns slice of Var names that were unused
// after a Transform run.
func (rv *RefVarTransformer) UnusedVars() []string {
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
func (rv *RefVarTransformer) Transform(m resmap.ResMap) error {
	rv.replacementCounts = make(map[string]int)
	rv.mappingFunc = expansion.MappingFuncFor(
		rv.replacementCounts, rv.varMap)
	for _, res := range m.Resources() {
		for _, fieldSpec := range rv.fieldSpecs {
			if res.OrgId().IsSelected(&fieldSpec.Gvk) {
				if err := MutateField(
					res.Map(), fieldSpec.PathSlice(),
					false, rv.replaceVars); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
