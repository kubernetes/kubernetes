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

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

// PruneDefaults prunes default values according to the schema and according to
// the ObjectMeta definition of the running server. It mutates the passed schema.
func PruneDefaults(s *structuralschema.Structural) error {
	_, err := pruner{s}.pruneDefaults(s, NewRootObjectFunc())
	return err
}

type pruner struct {
	rootSchema *structuralschema.Structural
}

func (p *pruner) pruneDefaults(s *structuralschema.Structural, f SurroundingObjectFunc) (bool, error) {
	if s == nil {
		return false, nil
	}

	if s.Default.Object != nil {
		obj, acc, err := f(s.Default.Object)
		if err != nil {
			return false, fmt.Errorf("failed to prune default value: %v", err)
		}
		pruning.Prune(obj, p.rootSchema, true)
		s.Default.Object, _, err = acc(obj)
		if err != nil {
			return false, fmt.Errorf("failed to prune default value: %v", err)
		}
	}

	if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
		if _, err := p.pruneDefaults(s.AdditionalProperties.Structural, f.Child("*")); err != nil {
			return false, err
		}
	}
	if s.Items != nil {
		if _, err := p.pruneDefaults(s.Items, f.Index()); err != nil {
			return false, err
		}
	}
	for k, subSchema := range s.Properties {
		if changed, err := p.pruneDefaults(&subSchema, f.Child(k)); err != nil {
			return false, err
		} else if changed {
			s.Properties[k] = subSchema
		}
	}

	return false, nil
}
