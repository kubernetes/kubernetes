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

	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"

	"k8s.io/code-generator/cmd/client-gen/generators/util"
)

// expansionGenerator produces a file for a expansion interfaces.
type expansionGenerator struct {
	generator.GoGenerator
	outputPath string
	types      []*types.Type
}

// We only want to call GenerateType() once per group.
func (g *expansionGenerator) Filter(c *generator.Context, t *types.Type) bool {
	return t == g.types[0]
}

func (g *expansionGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	for _, t := range g.types {
		tags := util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
		manualFile := filepath.Join(g.outputPath, strings.ToLower(t.Name.Name+"_expansion.go"))
		if _, err := os.Stat(manualFile); err == nil {
			klog.V(4).Infof("file %q exists, not generating", manualFile)
		} else if os.IsNotExist(err) {
			sw.Do(expansionInterfaceTemplate, t)
			if !tags.NonNamespaced {
				sw.Do(namespacedExpansionInterfaceTemplate, t)
			}
		} else {
			return err
		}
	}
	return sw.Error()
}

var expansionInterfaceTemplate = `
// $.|public$ListerExpansion allows custom methods to be added to
// $.|public$Lister.
type $.|public$ListerExpansion interface {}
`

var namespacedExpansionInterfaceTemplate = `
// $.|public$NamespaceListerExpansion allows custom methods to be added to
// $.|public$NamespaceLister.
type $.|public$NamespaceListerExpansion interface {}
`
