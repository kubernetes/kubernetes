package transformers

import (
	"fmt"

	"sigs.k8s.io/kustomize/pkg/expansion"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

type refvarTransformer struct {
	fieldSpecs []config.FieldSpec
	vars       map[string]string
}

// NewRefVarTransformer returns a Trasformer that replaces $(VAR) style variables with values.
func NewRefVarTransformer(vars map[string]string, p []config.FieldSpec) Transformer {
	return &refvarTransformer{
		vars:       vars,
		fieldSpecs: p,
	}
}

// Transform determines the final values of variables:
//
// 1.  Determine the final value of each variable:
//     a.  If the variable's Value is set, expand the `$(var)` references to other
//         variables in the .Value field; the sources of variables are the declared
//         variables of the container and the service environment variables
//     b.  If a source is defined for an environment variable, resolve the source
// 2.  Create the container's environment in the order variables are declared
// 3.  Add remaining service environment vars
func (rv *refvarTransformer) Transform(resources resmap.ResMap) error {
	for resId := range resources {
		objMap := resources[resId].Map()
		for _, pc := range rv.fieldSpecs {
			if !resId.Gvk().IsSelected(&pc.Gvk) {
				continue
			}
			err := mutateField(objMap, pc.PathSlice(), false, func(in interface{}) (interface{}, error) {
				var (
					mappingFunc = expansion.MappingFuncFor(rv.vars)
				)
				switch vt := in.(type) {
				case []interface{}:
					var xs []string
					for _, a := range in.([]interface{}) {
						xs = append(xs, expansion.Expand(a.(string), mappingFunc))
					}
					return xs, nil
				case interface{}:
					s, ok := in.(string)
					if !ok {
						return nil, fmt.Errorf("%#v is expected to be %T", in, s)
					}
					runtimeVal := expansion.Expand(s, mappingFunc)
					return runtimeVal, nil
				default:
					return "", fmt.Errorf("invalid type encountered %T", vt)
				}
			})
			if err != nil {
				return err
			}
		}
	}
	return nil
}
