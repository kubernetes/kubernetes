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

package validation

import (
	"fmt"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
)

// DeriveSkeleton returns the skeleton schema for the given OpenAPI validation schema.
// A skeleton schema
// - has not value validation (min, max, regex, etc.)
// - has no propositional construct (anyOf, allOf, oneOf, not)
// - is enough for pruning and defaulting
// - validates a given input if the full schema valides it.
//
// The result only contains `type`, `items`, `additionalItems`, `properties`,
// `patternProperties`, `additionalProperties`, `default`.
func DeriveSkeleton(v *apiextensions.JSONSchemaProps) (*apiextensions.JSONSchemaProps, error) {
	if v == nil {
		return nil, nil
	}

	// apply DeriveSkeleton to all propositional sub-schemata
	var propositionSchemas []*apiextensions.JSONSchemaProps
	for _, x := range v.AllOf {
		copy := x
		propositionSchemas = append(propositionSchemas, &copy)
	}
	for _, x := range v.AnyOf {
		copy := x
		propositionSchemas = append(propositionSchemas, &copy)
	}
	for _, x := range v.OneOf {
		copy := x
		propositionSchemas = append(propositionSchemas, &copy)
	}
	if v.Not != nil {
		propositionSchemas = append(propositionSchemas, v.Not)
	}
	for i, s := range propositionSchemas {
		s, err := DeriveSkeleton(s)
		if err != nil {
			return nil, err
		}
		propositionSchemas[i] = s
	}

	// merge with root
	return merge(append([]*apiextensions.JSONSchemaProps{drop(v)}, propositionSchemas...)...)
}

func drop(v *apiextensions.JSONSchemaProps) *apiextensions.JSONSchemaProps {
	ret := &apiextensions.JSONSchemaProps{
		Properties:         v.Properties,
		PatternProperties:  v.PatternProperties,
		Items:              v.Items,
		AdditionalItems:    v.AdditionalItems,
		Type:               v.Type,
		Default:            v.Default,
		XKubernetesNoPrune: v.XKubernetesNoPrune,
	}
	if v.AdditionalProperties != nil && (v.AdditionalProperties.Schema != nil || v.AdditionalProperties.Allows == false) {
		// ignore redundant "additionalProperties: true". We allow those inside CRD validation, but the complicate
		// the logic of skeleton generation.
		ret.AdditionalProperties = v.AdditionalProperties
	}
	return ret
}

func merge(xs ...*apiextensions.JSONSchemaProps) (*apiextensions.JSONSchemaProps, error) {
	if len(xs) == 0 {
		return nil, nil
	}

	ret := &apiextensions.JSONSchemaProps{}

	// type: t if all x_i agree on t as type, undefined otherwise
	first := true
	for _, x := range xs {
		if !first && ret.Type != x.Type {
			ret.Type = ""
			break
		}

		first = false
		ret.Type = x.Type
	}

	// properties: { k_i: merge(v_i1, …, v_ik) for keys appearing in x_ij with values x_ij }
	propertiesByKey := map[string][]*apiextensions.JSONSchemaProps{}
	for _, x := range xs {
		for k, v := range x.Properties {
			copy := v
			propertiesByKey[k] = append(propertiesByKey[k], &copy)
		}
	}
	for k, vs := range propertiesByKey {
		merged, err := merge(vs...)
		if err != nil {
			return nil, err
		}
		if merged != nil {
			if ret.Properties == nil {
				ret.Properties = map[string]apiextensions.JSONSchemaProps{}
			}
			ret.Properties[k] = *merged
		}
	}

	// patternProperties: { k_i: merge(v_i1, …, v_ik) for keys appearing in x_ij with values x_ij }
	propertiesByKey = map[string][]*apiextensions.JSONSchemaProps{}
	for _, x := range xs {
		for k, v := range x.PatternProperties {
			copy := v
			propertiesByKey[k] = append(propertiesByKey[k], &copy)
		}
	}
	for k, vs := range propertiesByKey {
		merged, err := merge(vs...)
		if err != nil {
			return nil, err
		}
		if merged != nil {
			// TODO: enable the following if we forbid properties and patternProperties in CRD validation
			//if ret.Properties != nil {
			//	return nil, fmt.Errorf("properties and patternProperties must not be set at the same time")
			//}
			if ret.PatternProperties == nil {
				ret.PatternProperties = map[string]apiextensions.JSONSchemaProps{}
			}
			ret.PatternProperties[k] = *merged
		}
	}

	// additionalProperties: merge(p_1, …, p_i), for all x_i with defined additionalProperties p_i
	var additionalProperties []*apiextensions.JSONSchemaProps
	for _, x := range xs {
		if x.AdditionalProperties == nil {
			continue
		}

		if x.AdditionalProperties.Schema == nil {
			// we weaken  apiextensions.JSONSchemaProps{Allows: false} to apiextensions.JSONSchemaProps{Allows: true}
			// i.e. this won't be verified in skeleton validation, but only in the full OpenAPI validation in the registry.
			additionalProperties = append(additionalProperties, &apiextensions.JSONSchemaProps{})
		} else {
			additionalProperties = append(additionalProperties, x.AdditionalProperties.Schema)
		}
	}
	if len(additionalProperties) > 0 {
		merged, err := merge(additionalProperties...)
		if err != nil {
			return nil, err
		}
		if merged != nil {
			if ret.Properties != nil {
				return nil, fmt.Errorf("properties and additionalProperties must not be set at the same time")
			}
			ret.AdditionalProperties = &apiextensions.JSONSchemaPropsOrBool{Schema: merged}
		}
	}

	// items: [ merge(x_i1, …, x_ik) ] where s_ij = s_j.items[i] if defined,
	var allItemSchemas []*apiextensions.JSONSchemaProps
	maxSize := 0
	for _, x := range xs {
		if x.Items == nil {
			continue
		}

		if x.Items.Schema != nil {
			allItemSchemas = append(allItemSchemas, x.Items.Schema)
			continue
		}

		if l := len(x.Items.JSONSchemas); l > maxSize {
			maxSize = l
		}
	}
	if maxSize == 0 && len(allItemSchemas) > 0 {
		merged, err := merge(allItemSchemas...)
		if err != nil {
			return nil, err
		}
		ret.Items = &apiextensions.JSONSchemaPropsOrArray{Schema: merged}
	} else if maxSize > 0 {
		items := make([]apiextensions.JSONSchemaProps, maxSize)
		for i := 0; i < maxSize; i++ {
			var iSchemas []*apiextensions.JSONSchemaProps
			for _, x := range xs {
				if x.Items == nil || i >= len(x.Items.JSONSchemas) {
					continue
				}
				iSchemas = append(iSchemas, &x.Items.JSONSchemas[i])
			}
			merged, err := merge(append(iSchemas, allItemSchemas...)...)
			if err != nil {
				return nil, err
			}
			items[i] = *merged
		}
		ret.Items = &apiextensions.JSONSchemaPropsOrArray{JSONSchemas: items}
	}

	// additionalItems: merge(p_1, …, p_i), for all x_i with defined additionalItems p_i
	var additionalItems []*apiextensions.JSONSchemaProps
	for _, x := range xs {
		if x.AdditionalItems == nil {
			continue
		}

		if x.AdditionalItems.Schema == nil {
			// we weaken  apiextensions.JSONSchemaProps{Allows: false} to apiextensions.JSONSchemaProps{Allows: true}
			// i.e. this won't be verified in skeleton validation, but only in the full OpenAPI validation in the registry.
			additionalItems = append(additionalItems, &apiextensions.JSONSchemaProps{})
		} else {
			additionalItems = append(additionalItems, x.AdditionalItems.Schema)
		}
	}
	if len(additionalItems) > 0 {
		merged, err := merge(additionalItems...)
		if err != nil {
			return nil, err
		}
		if merged != nil {
			ret.AdditionalItems = &apiextensions.JSONSchemaPropsOrBool{Schema: merged}
		}
	}

	// default: first default value of  `x_i` or fail
	for _, x := range xs {
		if x.Default == nil {
			continue
		}

		if ret.Default != nil {
			return nil, fmt.Errorf("only one default expected, found %v and %v", ret.Default, x.Default)
		}

		ret.Default = x.Default
	}

	// x-kuberentes-no-prune: true trumps over false
	for _, x := range xs {
		if x.XKubernetesNoPrune == nil {
			continue
		}
		ret.XKubernetesNoPrune = x.XKubernetesNoPrune
		if *x.XKubernetesNoPrune == true {
			break
		}
	}

	return ret, nil
}
