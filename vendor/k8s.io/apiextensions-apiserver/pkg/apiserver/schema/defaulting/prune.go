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

package defaulting

import (
	"fmt"
	"reflect"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	structuralobjectmeta "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/objectmeta"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
	"k8s.io/apimachinery/pkg/runtime"
)

// PruneDefaults prunes default values according to the schema and according to
// the ObjectMeta definition of the running server. It mutates the passed schema.
func PruneDefaults(s *structuralschema.Structural) error {
	p := pruner{s}
	_, err := p.pruneDefaults(s, NewRootObjectFunc())
	return err
}

type pruner struct {
	rootSchema *structuralschema.Structural
}

func (p *pruner) pruneDefaults(s *structuralschema.Structural, f SurroundingObjectFunc) (changed bool, err error) {
	if s == nil {
		return false, nil
	}

	if s.Default.Object != nil {
		orig := runtime.DeepCopyJSONValue(s.Default.Object)

		obj, acc, err := f(s.Default.Object)
		if err != nil {
			return false, fmt.Errorf("failed to prune default value: %v", err)
		}
		if err := structuralobjectmeta.Coerce(nil, obj, p.rootSchema, true, true); err != nil {
			return false, fmt.Errorf("failed to prune default value: %v", err)
		}
		pruning.Prune(obj, p.rootSchema, true)
		s.Default.Object, _, err = acc(obj)
		if err != nil {
			return false, fmt.Errorf("failed to prune default value: %v", err)
		}

		changed = changed || !reflect.DeepEqual(orig, s.Default.Object)
	}

	if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
		c, err := p.pruneDefaults(s.AdditionalProperties.Structural, f.Child("*"))
		if err != nil {
			return false, err
		}
		changed = changed || c
	}
	if s.Items != nil {
		c, err := p.pruneDefaults(s.Items, f.Index())
		if err != nil {
			return false, err
		}
		changed = changed || c
	}
	for k, subSchema := range s.Properties {
		c, err := p.pruneDefaults(&subSchema, f.Child(k))
		if err != nil {
			return false, err
		}
		if c {
			s.Properties[k] = subSchema
			changed = true
		}
	}

	return changed, nil
}
