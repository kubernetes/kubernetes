package transformers

import (
	"fmt"
	"sigs.k8s.io/kustomize/pkg/expansion"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

type RefVarTransformer struct {
	varMap            map[string]string
	replacementCounts map[string]int
	fieldSpecs        []config.FieldSpec
	mappingFunc       func(string) string
}

// NewRefVarTransformer returns a new RefVarTransformer
// that replaces $(VAR) style variables with values.
// The fieldSpecs are the places to look for occurrences of $(VAR).
func NewRefVarTransformer(
	varMap map[string]string, fs []config.FieldSpec) *RefVarTransformer {
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
		var xs []string
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
				return nil, fmt.Errorf("%#v is expected to be %T", v, s)
			}
			xs[k] = expansion.Expand(s, rv.mappingFunc)
		}
		return xs, nil
	case interface{}:
		s, ok := in.(string)
		if !ok {
			return nil, fmt.Errorf("%#v is expected to be %T", in, s)
		}
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
	for id, res := range m {
		for _, fieldSpec := range rv.fieldSpecs {
			if id.Gvk().IsSelected(&fieldSpec.Gvk) {
				if err := mutateField(
					res.Map(), fieldSpec.PathSlice(),
					false, rv.replaceVars); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
