// Copyright 2020 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// gnostic-complexity is a plugin that generates a complexity summary of an API.
package main

import (
	"encoding/json"
	"path/filepath"

	"github.com/golang/protobuf/proto"
	metrics "github.com/googleapis/gnostic/metrics"
	openapiv2 "github.com/googleapis/gnostic/openapiv2"
	openapiv3 "github.com/googleapis/gnostic/openapiv3"
	plugins "github.com/googleapis/gnostic/plugins"
)

// This is the main function for the plugin.
func main() {
	env, err := plugins.NewEnvironment()
	env.RespondAndExitIfError(err)

	var complexity *metrics.Complexity

	for _, model := range env.Request.Models {
		switch model.TypeUrl {
		case "openapi.v2.Document":
			documentv2 := &openapiv2.Document{}
			err = proto.Unmarshal(model.Value, documentv2)
			if err == nil {
				complexity = analyzeOpenAPIv2Document(documentv2)
			}
		case "openapi.v3.Document":
			documentv3 := &openapiv3.Document{}
			err = proto.Unmarshal(model.Value, documentv3)
			if err == nil {
				complexity = analyzeOpenAPIv3Document(documentv3)
			}
		}
	}

	if complexity != nil {
		// Return JSON-serialized output.
		file := &plugins.File{}
		file.Name = filepath.Join(filepath.Dir(env.Request.SourceName), "complexity.json")
		file.Data, err = json.MarshalIndent(complexity, "", "  ")
		env.RespondAndExitIfError(err)
		file.Data = append(file.Data, []byte("\n")...)
		env.Response.Files = append(env.Response.Files, file)

		// Return binary-serialized output.
		file2 := &plugins.File{}
		file2.Name = filepath.Join(filepath.Dir(env.Request.SourceName), "complexity.pb")
		file2.Data, err = proto.Marshal(complexity)
		env.RespondAndExitIfError(err)
		env.Response.Files = append(env.Response.Files, file2)
	}

	env.RespondAndExit()
}

func newComplexity() *metrics.Complexity {
	return &metrics.Complexity{}
}

func analyzeOpenAPIv2Document(document *openapiv2.Document) *metrics.Complexity {
	summary := newComplexity()

	if document.Definitions != nil && document.Definitions.AdditionalProperties != nil {
		for _, pair := range document.Definitions.AdditionalProperties {
			analyzeSchema(summary, pair.Value)
		}
	}

	for _, pair := range document.Paths.Path {
		summary.PathCount++
		v := pair.Value
		if v.Get != nil {
			summary.GetCount++
		}
		if v.Post != nil {
			summary.PostCount++
		}
		if v.Put != nil {
			summary.PutCount++
		}
		if v.Delete != nil {
			summary.DeleteCount++
		}
	}
	return summary
}

func analyzeSchema(summary *metrics.Complexity, schema *openapiv2.Schema) {
	summary.SchemaCount++
	if schema.Properties != nil {
		for _, pair := range schema.Properties.AdditionalProperties {
			summary.SchemaPropertyCount++
			analyzeSchema(summary, pair.Value)
		}
	}
}

func analyzeOpenAPIv3Document(document *openapiv3.Document) *metrics.Complexity {
	summary := newComplexity()

	if document.Components != nil && document.Components.Schemas != nil {
		for _, pair := range document.Components.Schemas.AdditionalProperties {
			analyzeOpenAPIv3Schema(summary, pair.Value)
		}
	}

	for _, pair := range document.Paths.Path {
		summary.PathCount++
		v := pair.Value
		if v.Get != nil {
			summary.GetCount++
		}
		if v.Post != nil {
			summary.PostCount++
		}
		if v.Put != nil {
			summary.PutCount++
		}
		if v.Delete != nil {
			summary.DeleteCount++
		}
	}
	return summary
}

func analyzeOpenAPIv3Schema(summary *metrics.Complexity, schemaOrReference *openapiv3.SchemaOrReference) {
	summary.SchemaCount++
	schema := schemaOrReference.GetSchema()
	if schema != nil && schema.Properties != nil {
		for _, pair := range schema.Properties.AdditionalProperties {
			summary.SchemaPropertyCount++
			analyzeOpenAPIv3Schema(summary, pair.Value)
		}
	}
}
