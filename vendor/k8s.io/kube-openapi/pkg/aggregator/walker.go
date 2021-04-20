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
	"strings"

	"k8s.io/kube-openapi/pkg/validation/spec"
)

const (
	definitionPrefix = "#/definitions/"
)

// Run a readonlyReferenceWalker method on all references of an OpenAPI spec
type readonlyReferenceWalker struct {
	// walkRefCallback will be called on each reference. The input will never be nil.
	walkRefCallback func(ref *spec.Ref)

	// The spec to walk through.
	root *spec.Swagger
}

// walkOnAllReferences recursively walks on all references, while following references into definitions.
// it calls walkRef on each found reference.
func walkOnAllReferences(walkRef func(ref *spec.Ref), root *spec.Swagger) {
	alreadyVisited := map[string]bool{}

	walker := &readonlyReferenceWalker{
		root: root,
	}
	walker.walkRefCallback = func(ref *spec.Ref) {
		walkRef(ref)

		refStr := ref.String()
		if refStr == "" || !strings.HasPrefix(refStr, definitionPrefix) {
			return
		}
		defName := refStr[len(definitionPrefix):]

		if _, found := root.Definitions[defName]; found && !alreadyVisited[refStr] {
			alreadyVisited[refStr] = true
			def := root.Definitions[defName]
			walker.walkSchema(&def)
		}
	}
	walker.Start()
}

func (s *readonlyReferenceWalker) walkSchema(schema *spec.Schema) {
	if schema == nil {
		return
	}
	s.walkRefCallback(&schema.Ref)
	var v *spec.Schema
	if len(schema.Definitions)+len(schema.Properties)+len(schema.PatternProperties) > 0 {
		v = &spec.Schema{}
	}
	for k := range schema.Definitions {
		*v = schema.Definitions[k]
		s.walkSchema(v)
	}
	for k := range schema.Properties {
		*v = schema.Properties[k]
		s.walkSchema(v)
	}
	for k := range schema.PatternProperties {
		*v = schema.PatternProperties[k]
		s.walkSchema(v)
	}
	for i := range schema.AllOf {
		s.walkSchema(&schema.AllOf[i])
	}
	for i := range schema.AnyOf {
		s.walkSchema(&schema.AnyOf[i])
	}
	for i := range schema.OneOf {
		s.walkSchema(&schema.OneOf[i])
	}
	if schema.Not != nil {
		s.walkSchema(schema.Not)
	}
	if schema.AdditionalProperties != nil && schema.AdditionalProperties.Schema != nil {
		s.walkSchema(schema.AdditionalProperties.Schema)
	}
	if schema.AdditionalItems != nil && schema.AdditionalItems.Schema != nil {
		s.walkSchema(schema.AdditionalItems.Schema)
	}
	if schema.Items != nil {
		if schema.Items.Schema != nil {
			s.walkSchema(schema.Items.Schema)
		}
		for i := range schema.Items.Schemas {
			s.walkSchema(&schema.Items.Schemas[i])
		}
	}
}

func (s *readonlyReferenceWalker) walkParams(params []spec.Parameter) {
	if params == nil {
		return
	}
	for _, param := range params {
		s.walkRefCallback(&param.Ref)
		s.walkSchema(param.Schema)
		if param.Items != nil {
			s.walkRefCallback(&param.Items.Ref)
		}
	}
}

func (s *readonlyReferenceWalker) walkResponse(resp *spec.Response) {
	if resp == nil {
		return
	}
	s.walkRefCallback(&resp.Ref)
	s.walkSchema(resp.Schema)
}

func (s *readonlyReferenceWalker) walkOperation(op *spec.Operation) {
	if op == nil {
		return
	}
	s.walkParams(op.Parameters)
	if op.Responses == nil {
		return
	}
	s.walkResponse(op.Responses.Default)
	for _, r := range op.Responses.StatusCodeResponses {
		s.walkResponse(&r)
	}
}

func (s *readonlyReferenceWalker) Start() {
	if s.root.Paths == nil {
		return
	}
	for _, pathItem := range s.root.Paths.Paths {
		s.walkParams(pathItem.Parameters)
		s.walkOperation(pathItem.Delete)
		s.walkOperation(pathItem.Get)
		s.walkOperation(pathItem.Head)
		s.walkOperation(pathItem.Options)
		s.walkOperation(pathItem.Patch)
		s.walkOperation(pathItem.Post)
		s.walkOperation(pathItem.Put)
	}
}
