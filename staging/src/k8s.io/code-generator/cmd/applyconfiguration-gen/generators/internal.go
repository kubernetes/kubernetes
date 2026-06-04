/*
Copyright 2021 The Kubernetes Authors.

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

package generators

import (
	"io"

	yaml "go.yaml.in/yaml/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/kube-openapi/pkg/schemaconv"
)

// utilGenerator generates the ForKind() utility function.
type internalGenerator struct {
	generator.GoGenerator
	outputPackage string
	imports       namer.ImportTracker
	typeModels    *typeModels
	filtered      bool
}

var _ generator.Generator = &internalGenerator{}

func (g *internalGenerator) Filter(*generator.Context, *types.Type) bool {
	// generate file exactly once
	if !g.filtered {
		g.filtered = true
		return true
	}
	return false
}

func (g *internalGenerator) Namers(*generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw":          namer.NewRawNamer(g.outputPackage, g.imports),
		"singularKind": namer.NewPublicNamer(0),
	}
}

func (g *internalGenerator) Imports(*generator.Context) (imports []string) {
	return g.imports.ImportLines()
}

func (g *internalGenerator) GenerateType(c *generator.Context, _ *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "{{", "}}")

	schema, err := schemaconv.ToSchema(g.typeModels.models)
	if err != nil {
		return err
	}
	schemaYAML, err := yaml.Marshal(schema)
	if err != nil {
		return err
	}
	sw.Do(schemaBlock, map[string]interface{}{
		"schemaYAML":   string(schemaYAML),
		"smdParser":    smdParser,
		"smdNewParser": smdNewParser,
		"fmtSprintf":   fmtSprintf,
		"syncOnce":     syncOnce,
		"yamlObject":   yamlObject,
	})

	return sw.Error()
}

var schemaBlock = `
func Parser() *{{.smdParser|raw}} {
	parserOnce.Do(func() {
		var err error
		parser, err = {{.smdNewParser|raw}}(schemaYAML)
		if err != nil {
			panic({{.fmtSprintf|raw}}("Failed to parse schema: %v", err))
		}
	})
	return parser
}

var parserOnce {{.syncOnce|raw}}
var parser *{{.smdParser|raw}}
var schemaYAML = {{.yamlObject|raw}}(` + "`{{.schemaYAML}}`" + `)
`
