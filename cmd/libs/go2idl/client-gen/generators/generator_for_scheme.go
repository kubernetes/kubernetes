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
	"fmt"
	"io"
	"strings"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
	clientgentypes "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/types"
)

// genScheme produces a package for a clientset with the scheme, codecs and parameter codecs.
type genScheme struct {
	generator.DefaultGen
	outputPackage   string
	groups          []clientgentypes.GroupVersions
	inputPaths      map[clientgentypes.GroupVersion]string
	imports         namer.ImportTracker
	schemeGenerated bool
}

func (g *genScheme) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

// We only want to call GenerateType() once.
func (g *genScheme) Filter(c *generator.Context, t *types.Type) bool {
	ret := !g.schemeGenerated
	g.schemeGenerated = true
	return ret
}

func (g *genScheme) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	for _, group := range g.groups {
		for _, version := range group.Versions {
			inputPath := g.inputPaths[clientgentypes.GroupVersion{Group: group.Group, Version: version}]
			imports = append(imports, strings.ToLower(fmt.Sprintf("%s%s \"%s\"", group.Group.NonEmpty(), version.NonEmpty(), inputPath)))
		}
	}
	return
}

func (g *genScheme) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	allGroups := clientgentypes.ToGroupVersionPackages(g.groups)

	m := map[string]interface{}{
		"allGroups":                 allGroups,
		"runtimeNewParameterCodec":  c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/runtime", Name: "NewParameterCodec"}),
		"runtimeNewScheme":          c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/runtime", Name: "NewScheme"}),
		"serializerNewCodecFactory": c.Universe.Function(types.Name{Package: "k8s.io/apimachinery/pkg/runtime/serializer", Name: "NewCodecFactory"}),
	}
	sw.Do(globalsTemplate, m)

	return sw.Error()
}

var globalsTemplate = `
var Scheme = $.runtimeNewScheme|raw$()
var Codecs = $.serializerNewCodecFactory|raw$(Scheme)
var ParameterCodec = $.runtimeNewParameterCodec|raw$(Scheme)

func init() {
	$range .allGroups$ $.InputPackageName$.AddToScheme(Scheme)
	$end$
}
`
