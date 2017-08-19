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

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

// versionInterfaceGenerator generates the per-version interface file.
type versionInterfaceGenerator struct {
	generator.DefaultGen
	outputPackage             string
	imports                   namer.ImportTracker
	types                     []*types.Type
	filtered                  bool
	internalInterfacesPackage string
}

var _ generator.Generator = &versionInterfaceGenerator{}

func (g *versionInterfaceGenerator) Filter(c *generator.Context, t *types.Type) bool {
	if !g.filtered {
		g.filtered = true
		return true
	}
	return false
}

func (g *versionInterfaceGenerator) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *versionInterfaceGenerator) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	return
}

func (g *versionInterfaceGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	m := map[string]interface{}{
		"interfacesSharedInformerFactory": c.Universe.Type(types.Name{Package: g.internalInterfacesPackage, Name: "SharedInformerFactory"}),
		"types": g.types,
	}

	sw.Do(versionTemplate, m)

	return sw.Error()
}

var versionTemplate = `
// Interface provides access to all the informers in this group version.
type Interface interface {
	$range .types -$
		// $.|publicPlural$ returns a $.|public$Informer.
		$.|publicPlural$() $.|public$Informer
	$end$
}

type version struct {
	$.interfacesSharedInformerFactory|raw$
}

// New returns a new Interface.
func New(f $.interfacesSharedInformerFactory|raw$) Interface {
	return &version{f}
}

$range .types$
// $.|publicPlural$ returns a $.|public$Informer.
func (v *version) $.|publicPlural$() $.|public$Informer {
	return &$.|private$Informer{factory: v.SharedInformerFactory}
}
$end$
`
