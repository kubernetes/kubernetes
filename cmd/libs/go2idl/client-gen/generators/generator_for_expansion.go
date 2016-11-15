/*
Copyright 2016 The Kubernetes Authors.

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
	"os"
	"path/filepath"
	"strings"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/types"
)

// genExpansion produces a file for a group client, e.g. ExtensionsClient for the extension group.
type genExpansion struct {
	generator.DefaultGen
	groupPath string
	// types in a group
	types []*types.Type
}

// We only want to call GenerateType() once per group.
func (g *genExpansion) Filter(c *generator.Context, t *types.Type) bool {
	return t == g.types[0]
}

func (g *genExpansion) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	for _, t := range g.types {
		if _, err := os.Stat(filepath.Join(g.groupPath, strings.ToLower(t.Name.Name+"_expansion.go"))); os.IsNotExist(err) {
			sw.Do(expansionInterfaceTemplate, t)
		}
	}
	return sw.Error()
}

var expansionInterfaceTemplate = `
type $.|public$Expansion interface {}
`
