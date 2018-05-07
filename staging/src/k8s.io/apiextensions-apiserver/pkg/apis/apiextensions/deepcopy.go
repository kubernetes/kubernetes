/*
Copyright 2017 The Kubernetes Authors.

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

package apiextensions

import "k8s.io/apimachinery/pkg/runtime"

// TODO: Update this after a tag is created for interface fields in DeepCopy
func (in *JSONSchemaProps) DeepCopy() *JSONSchemaProps {
	if in == nil {
		return nil
	}
	out := new(JSONSchemaProps)

	*out = *in

	if in.Default != nil {
		defaultJSON := JSON(runtime.DeepCopyJSONValue(*(in.Default)))
		out.Default = &(defaultJSON)
	} else {
		out.Default = nil
	}

	if in.Example != nil {
		exampleJSON := JSON(runtime.DeepCopyJSONValue(*(in.Example)))
		out.Example = &(exampleJSON)
	} else {
		out.Example = nil
	}

	if in.Ref != nil {
		in, out := &in.Ref, &out.Ref
		if *in == nil {
			*out = nil
		} else {
			*out = new(string)
			**out = **in
		}
	}

	if in.Maximum != nil {
		in, out := &in.Maximum, &out.Maximum
		if *in == nil {
			*out = nil
		} else {
			*out = new(float64)
			**out = **in
		}
	}

	if in.Minimum != nil {
		in, out := &in.Minimum, &out.Minimum
		if *in == nil {
			*out = nil
		} else {
			*out = new(float64)
			**out = **in
		}
	}

	if in.MaxLength != nil {
		in, out := &in.MaxLength, &out.MaxLength
		if *in == nil {
			*out = nil
		} else {
			*out = new(int64)
			**out = **in
		}
	}

	if in.MinLength != nil {
		in, out := &in.MinLength, &out.MinLength
		if *in == nil {
			*out = nil
		} else {
			*out = new(int64)
			**out = **in
		}
	}
	if in.MaxItems != nil {
		in, out := &in.MaxItems, &out.MaxItems
		if *in == nil {
			*out = nil
		} else {
			*out = new(int64)
			**out = **in
		}
	}

	if in.MinItems != nil {
		in, out := &in.MinItems, &out.MinItems
		if *in == nil {
			*out = nil
		} else {
			*out = new(int64)
			**out = **in
		}
	}

	if in.MultipleOf != nil {
		in, out := &in.MultipleOf, &out.MultipleOf
		if *in == nil {
			*out = nil
		} else {
			*out = new(float64)
			**out = **in
		}
	}

	if in.Enum != nil {
		out.Enum = make([]JSON, len(in.Enum))
		for i := range in.Enum {
			out.Enum[i] = runtime.DeepCopyJSONValue(in.Enum[i])
		}
	}

	if in.MaxProperties != nil {
		in, out := &in.MaxProperties, &out.MaxProperties
		if *in == nil {
			*out = nil
		} else {
			*out = new(int64)
			**out = **in
		}
	}

	if in.MinProperties != nil {
		in, out := &in.MinProperties, &out.MinProperties
		if *in == nil {
			*out = nil
		} else {
			*out = new(int64)
			**out = **in
		}
	}

	if in.Required != nil {
		in, out := &in.Required, &out.Required
		*out = make([]string, len(*in))
		copy(*out, *in)
	}

	if in.Items != nil {
		in, out := &in.Items, &out.Items
		if *in == nil {
			*out = nil
		} else {
			*out = new(JSONSchemaPropsOrArray)
			(*in).DeepCopyInto(*out)
		}
	}

	if in.AllOf != nil {
		in, out := &in.AllOf, &out.AllOf
		*out = make([]JSONSchemaProps, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}

	if in.OneOf != nil {
		in, out := &in.OneOf, &out.OneOf
		*out = make([]JSONSchemaProps, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}
	if in.AnyOf != nil {
		in, out := &in.AnyOf, &out.AnyOf
		*out = make([]JSONSchemaProps, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}

	if in.Not != nil {
		in, out := &in.Not, &out.Not
		if *in == nil {
			*out = nil
		} else {
			*out = new(JSONSchemaProps)
			(*in).DeepCopyInto(*out)
		}
	}

	if in.Properties != nil {
		in, out := &in.Properties, &out.Properties
		*out = make(map[string]JSONSchemaProps, len(*in))
		for key, val := range *in {
			(*out)[key] = *val.DeepCopy()
		}
	}

	if in.AdditionalProperties != nil {
		in, out := &in.AdditionalProperties, &out.AdditionalProperties
		if *in == nil {
			*out = nil
		} else {
			*out = new(JSONSchemaPropsOrBool)
			(*in).DeepCopyInto(*out)
		}
	}

	if in.PatternProperties != nil {
		in, out := &in.PatternProperties, &out.PatternProperties
		*out = make(map[string]JSONSchemaProps, len(*in))
		for key, val := range *in {
			(*out)[key] = *val.DeepCopy()
		}
	}

	if in.Dependencies != nil {
		in, out := &in.Dependencies, &out.Dependencies
		*out = make(JSONSchemaDependencies, len(*in))
		for key, val := range *in {
			(*out)[key] = *val.DeepCopy()
		}
	}

	if in.AdditionalItems != nil {
		in, out := &in.AdditionalItems, &out.AdditionalItems
		if *in == nil {
			*out = nil
		} else {
			*out = new(JSONSchemaPropsOrBool)
			(*in).DeepCopyInto(*out)
		}
	}

	if in.Definitions != nil {
		in, out := &in.Definitions, &out.Definitions
		*out = make(JSONSchemaDefinitions, len(*in))
		for key, val := range *in {
			(*out)[key] = *val.DeepCopy()
		}
	}

	if in.ExternalDocs != nil {
		in, out := &in.ExternalDocs, &out.ExternalDocs
		if *in == nil {
			*out = nil
		} else {
			*out = new(ExternalDocumentation)
			(*in).DeepCopyInto(*out)
		}
	}

	return out
}
