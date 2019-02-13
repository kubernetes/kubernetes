package transformers

import (
	"fmt"

	"sigs.k8s.io/kustomize/pkg/expansion"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

type refvarTransformer struct {
	fieldSpecs  []config.FieldSpec
	mappingFunc func(string) string
}

// NewRefVarTransformer returns a Transformer that replaces $(VAR) style
// variables with values.
// The fieldSpecs are the places to look for occurrences of $(VAR).
func NewRefVarTransformer(
	varMap map[string]string, fs []config.FieldSpec) Transformer {
	if len(varMap) == 0 {
		return NewNoOpTransformer()
	}
	return &refvarTransformer{
		fieldSpecs:  fs,
		mappingFunc: expansion.MappingFuncFor(varMap),
	}
}

// replaceVars accepts as 'in' a string, or string array, which can have
// embedded instances of $VAR style variables, e.g. a container command string.
// The function returns the string with the variables expanded to their final
// values.
func (rv *refvarTransformer) replaceVars(in interface{}) (interface{}, error) {
	switch vt := in.(type) {
	case []interface{}:
		var xs []string
		for _, a := range in.([]interface{}) {
			xs = append(xs, expansion.Expand(a.(string), rv.mappingFunc))
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

// Transform replaces $(VAR) style variables with values.
func (rv *refvarTransformer) Transform(m resmap.ResMap) error {
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
