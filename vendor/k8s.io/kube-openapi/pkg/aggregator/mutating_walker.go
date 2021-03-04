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

package aggregator

import (
	_ "net/http/pprof"

	"github.com/go-openapi/spec"
)

// Run a walkRefCallback method on all references of an OpenAPI spec, replacing the values.
type mutatingReferenceWalker struct {
	// walkRefCallback will be called on each reference. Do not mutate the input, always create a copy first and return that.
	walkRefCallback func(ref *spec.Ref) *spec.Ref
}

// replaceReferences rewrites the references without mutating the input.
// The output might share data with the input.
func replaceReferences(walkRef func(ref *spec.Ref) *spec.Ref, sp *spec.Swagger) *spec.Swagger {
	walker := &mutatingReferenceWalker{walkRefCallback: walkRef}
	return walker.Start(sp)
}

func (w *mutatingReferenceWalker) walkSchema(schema *spec.Schema) *spec.Schema {
	if schema == nil {
		return nil
	}

	orig := schema
	clone := func() {
		if orig == schema {
			schema = &spec.Schema{}
			*schema = *orig
		}
	}

	if r := w.walkRefCallback(&schema.Ref); r != &schema.Ref {
		clone()
		schema.Ref = *r
	}

	definitionsCloned := false
	for k, v := range schema.Definitions {
		if s := w.walkSchema(&v); s != &v {
			if !definitionsCloned {
				definitionsCloned = true
				clone()
				schema.Definitions = make(spec.Definitions, len(orig.Definitions))
				for k2, v2 := range orig.Definitions {
					schema.Definitions[k2] = v2
				}
			}
			schema.Definitions[k] = *s
		}
	}

	propertiesCloned := false
	for k, v := range schema.Properties {
		if s := w.walkSchema(&v); s != &v {
			if !propertiesCloned {
				propertiesCloned = true
				clone()
				schema.Properties = make(map[string]spec.Schema, len(orig.Properties))
				for k2, v2 := range orig.Properties {
					schema.Properties[k2] = v2
				}
			}
			schema.Properties[k] = *s
		}
	}

	patternPropertiesCloned := false
	for k, v := range schema.PatternProperties {
		if s := w.walkSchema(&v); s != &v {
			if !patternPropertiesCloned {
				patternPropertiesCloned = true
				clone()
				schema.PatternProperties = make(map[string]spec.Schema, len(orig.PatternProperties))
				for k2, v2 := range orig.PatternProperties {
					schema.PatternProperties[k2] = v2
				}
			}
			schema.PatternProperties[k] = *s
		}
	}

	allOfCloned := false
	for i := range schema.AllOf {
		if s := w.walkSchema(&schema.AllOf[i]); s != &schema.AllOf[i] {
			if !allOfCloned {
				allOfCloned = true
				clone()
				schema.AllOf = make([]spec.Schema, len(orig.AllOf))
				copy(schema.AllOf, orig.AllOf)
			}
			schema.AllOf[i] = *s
		}
	}

	anyOfCloned := false
	for i := range schema.AnyOf {
		if s := w.walkSchema(&schema.AnyOf[i]); s != &schema.AnyOf[i] {
			if !anyOfCloned {
				anyOfCloned = true
				clone()
				schema.AnyOf = make([]spec.Schema, len(orig.AnyOf))
				copy(schema.AnyOf, orig.AnyOf)
			}
			schema.AnyOf[i] = *s
		}
	}

	oneOfCloned := false
	for i := range schema.OneOf {
		if s := w.walkSchema(&schema.OneOf[i]); s != &schema.OneOf[i] {
			if !oneOfCloned {
				oneOfCloned = true
				clone()
				schema.OneOf = make([]spec.Schema, len(orig.OneOf))
				copy(schema.OneOf, orig.OneOf)
			}
			schema.OneOf[i] = *s
		}
	}

	if schema.Not != nil {
		if s := w.walkSchema(schema.Not); s != schema.Not {
			clone()
			schema.Not = s
		}
	}

	if schema.AdditionalProperties != nil && schema.AdditionalProperties.Schema != nil {
		if s := w.walkSchema(schema.AdditionalProperties.Schema); s != schema.AdditionalProperties.Schema {
			clone()
			schema.AdditionalProperties = &spec.SchemaOrBool{Schema: s, Allows: schema.AdditionalProperties.Allows}
		}
	}

	if schema.AdditionalItems != nil && schema.AdditionalItems.Schema != nil {
		if s := w.walkSchema(schema.AdditionalItems.Schema); s != schema.AdditionalItems.Schema {
			clone()
			schema.AdditionalItems = &spec.SchemaOrBool{Schema: s, Allows: schema.AdditionalItems.Allows}
		}
	}

	if schema.Items != nil {
		if schema.Items.Schema != nil {
			if s := w.walkSchema(schema.Items.Schema); s != schema.Items.Schema {
				clone()
				schema.Items = &spec.SchemaOrArray{Schema: s}
			}
		} else {
			itemsCloned := false
			for i := range schema.Items.Schemas {
				if s := w.walkSchema(&schema.Items.Schemas[i]); s != &schema.Items.Schemas[i] {
					if !itemsCloned {
						clone()
						schema.Items = &spec.SchemaOrArray{
							Schemas: make([]spec.Schema, len(orig.Items.Schemas)),
						}
						itemsCloned = true
						copy(schema.Items.Schemas, orig.Items.Schemas)
					}
					schema.Items.Schemas[i] = *s
				}
			}
		}
	}

	return schema
}

func (w *mutatingReferenceWalker) walkParameter(param *spec.Parameter) *spec.Parameter {
	if param == nil {
		return nil
	}

	orig := param
	cloned := false
	clone := func() {
		if !cloned {
			cloned = true
			param = &spec.Parameter{}
			*param = *orig
		}
	}

	if r := w.walkRefCallback(&param.Ref); r != &param.Ref {
		clone()
		param.Ref = *r
	}
	if s := w.walkSchema(param.Schema); s != param.Schema {
		clone()
		param.Schema = s
	}
	if param.Items != nil {
		if r := w.walkRefCallback(&param.Items.Ref); r != &param.Items.Ref {
			param.Items.Ref = *r
		}
	}

	return param
}

func (w *mutatingReferenceWalker) walkParameters(params []spec.Parameter) ([]spec.Parameter, bool) {
	if params == nil {
		return nil, false
	}

	orig := params
	cloned := false
	clone := func() {
		if !cloned {
			cloned = true
			params = make([]spec.Parameter, len(params))
			copy(params, orig)
		}
	}

	for i := range params {
		if s := w.walkParameter(&params[i]); s != &params[i] {
			clone()
			params[i] = *s
		}
	}

	return params, cloned
}

func (w *mutatingReferenceWalker) walkResponse(resp *spec.Response) *spec.Response {
	if resp == nil {
		return nil
	}

	orig := resp
	cloned := false
	clone := func() {
		if !cloned {
			cloned = true
			resp = &spec.Response{}
			*resp = *orig
		}
	}

	if r := w.walkRefCallback(&resp.Ref); r != &resp.Ref {
		clone()
		resp.Ref = *r
	}
	if s := w.walkSchema(resp.Schema); s != resp.Schema {
		clone()
		resp.Schema = s
	}

	return resp
}

func (w *mutatingReferenceWalker) walkResponses(resps *spec.Responses) *spec.Responses {
	if resps == nil {
		return nil
	}

	orig := resps
	cloned := false
	clone := func() {
		if !cloned {
			cloned = true
			resps = &spec.Responses{}
			*resps = *orig
		}
	}

	if r := w.walkResponse(resps.ResponsesProps.Default); r != resps.ResponsesProps.Default {
		clone()
		resps.Default = r
	}

	responsesCloned := false
	for k, v := range resps.ResponsesProps.StatusCodeResponses {
		if r := w.walkResponse(&v); r != &v {
			if !responsesCloned {
				responsesCloned = true
				clone()
				resps.ResponsesProps.StatusCodeResponses = make(map[int]spec.Response, len(orig.StatusCodeResponses))
				for k2, v2 := range orig.StatusCodeResponses {
					resps.ResponsesProps.StatusCodeResponses[k2] = v2
				}
			}
			resps.ResponsesProps.StatusCodeResponses[k] = *r
		}
	}

	return resps
}

func (w *mutatingReferenceWalker) walkOperation(op *spec.Operation) *spec.Operation {
	if op == nil {
		return nil
	}

	orig := op
	cloned := false
	clone := func() {
		if !cloned {
			cloned = true
			op = &spec.Operation{}
			*op = *orig
		}
	}

	parametersCloned := false
	for i := range op.Parameters {
		if s := w.walkParameter(&op.Parameters[i]); s != &op.Parameters[i] {
			if !parametersCloned {
				parametersCloned = true
				clone()
				op.Parameters = make([]spec.Parameter, len(orig.Parameters))
				copy(op.Parameters, orig.Parameters)
			}
			op.Parameters[i] = *s
		}
	}

	if r := w.walkResponses(op.Responses); r != op.Responses {
		clone()
		op.Responses = r
	}

	return op
}

func (w *mutatingReferenceWalker) walkPathItem(pathItem *spec.PathItem) *spec.PathItem {
	if pathItem == nil {
		return nil
	}

	orig := pathItem
	cloned := false
	clone := func() {
		if !cloned {
			cloned = true
			pathItem = &spec.PathItem{}
			*pathItem = *orig
		}
	}

	if p, changed := w.walkParameters(pathItem.Parameters); changed {
		clone()
		pathItem.Parameters = p
	}
	if op := w.walkOperation(pathItem.Get); op != pathItem.Get {
		clone()
		pathItem.Get = op
	}
	if op := w.walkOperation(pathItem.Head); op != pathItem.Head {
		clone()
		pathItem.Head = op
	}
	if op := w.walkOperation(pathItem.Delete); op != pathItem.Delete {
		clone()
		pathItem.Delete = op
	}
	if op := w.walkOperation(pathItem.Options); op != pathItem.Options {
		clone()
		pathItem.Options = op
	}
	if op := w.walkOperation(pathItem.Patch); op != pathItem.Patch {
		clone()
		pathItem.Patch = op
	}
	if op := w.walkOperation(pathItem.Post); op != pathItem.Post {
		clone()
		pathItem.Post = op
	}
	if op := w.walkOperation(pathItem.Put); op != pathItem.Put {
		clone()
		pathItem.Put = op
	}

	return pathItem
}

func (w *mutatingReferenceWalker) walkPaths(paths *spec.Paths) *spec.Paths {
	if paths == nil {
		return nil
	}

	orig := paths
	cloned := false
	clone := func() {
		if !cloned {
			cloned = true
			paths = &spec.Paths{}
			*paths = *orig
		}
	}

	pathsCloned := false
	for k, v := range paths.Paths {
		if p := w.walkPathItem(&v); p != &v {
			if !pathsCloned {
				pathsCloned = true
				clone()
				paths.Paths = make(map[string]spec.PathItem, len(orig.Paths))
				for k2, v2 := range orig.Paths {
					paths.Paths[k2] = v2
				}
			}
			paths.Paths[k] = *p
		}
	}

	return paths
}

func (w *mutatingReferenceWalker) Start(swagger *spec.Swagger) *spec.Swagger {
	if swagger == nil {
		return nil
	}

	orig := swagger
	cloned := false
	clone := func() {
		if !cloned {
			cloned = true
			swagger = &spec.Swagger{}
			*swagger = *orig
		}
	}

	parametersCloned := false
	for k, v := range swagger.Parameters {
		if p := w.walkParameter(&v); p != &v {
			if !parametersCloned {
				parametersCloned = true
				clone()
				swagger.Parameters = make(map[string]spec.Parameter, len(orig.Parameters))
				for k2, v2 := range orig.Parameters {
					swagger.Parameters[k2] = v2
				}
			}
			swagger.Parameters[k] = *p
		}
	}

	responsesCloned := false
	for k, v := range swagger.Responses {
		if r := w.walkResponse(&v); r != &v {
			if !responsesCloned {
				responsesCloned = true
				clone()
				swagger.Responses = make(map[string]spec.Response, len(orig.Responses))
				for k2, v2 := range orig.Responses {
					swagger.Responses[k2] = v2
				}
			}
			swagger.Responses[k] = *r
		}
	}

	definitionsCloned := false
	for k, v := range swagger.Definitions {
		if s := w.walkSchema(&v); s != &v {
			if !definitionsCloned {
				definitionsCloned = true
				clone()
				swagger.Definitions = make(spec.Definitions, len(orig.Definitions))
				for k2, v2 := range orig.Definitions {
					swagger.Definitions[k2] = v2
				}
			}
			swagger.Definitions[k] = *s
		}
	}

	if swagger.Paths != nil {
		if p := w.walkPaths(swagger.Paths); p != swagger.Paths {
			clone()
			swagger.Paths = p
		}
	}

	return swagger
}
