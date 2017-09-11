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

	"github.com/golang/glog"
)

// factoryInterfaceGenerator produces a file of interfaces used to break a dependency cycle for
// informer registration
type factoryInterfaceGenerator struct {
	generator.DefaultGen
	outputPackage    string
	imports          namer.ImportTracker
	clientSetPackage string
	filtered         bool
}

var _ generator.Generator = &factoryInterfaceGenerator{}

func (g *factoryInterfaceGenerator) Filter(c *generator.Context, t *types.Type) bool {
	if !g.filtered {
		g.filtered = true
		return true
	}
	return false
}

func (g *factoryInterfaceGenerator) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *factoryInterfaceGenerator) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	return
}

func (g *factoryInterfaceGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "{{", "}}")

	glog.V(5).Infof("processing type %v", t)

	m := map[string]interface{}{
		"cacheSharedIndexInformer": c.Universe.Type(cacheSharedIndexInformer),
		"clientSetPackage":         c.Universe.Type(types.Name{Package: g.clientSetPackage, Name: "Interface"}),
		"runtimeObject":            c.Universe.Type(runtimeObject),
		"timeDuration":             c.Universe.Type(timeDuration),
	}

	sw.Do(externalSharedInformerFactoryInterface, m)

	return sw.Error()
}

var externalSharedInformerFactoryInterface = `
type NewInformerFunc func({{.clientSetPackage|raw}}, {{.timeDuration|raw}}) cache.SharedIndexInformer

// SharedInformerFactory a small interface to allow for adding an informer without an import cycle
type SharedInformerFactory interface {
	Start(stopCh <-chan struct{})
	InformerFor(obj {{.runtimeObject|raw}}, newFunc NewInformerFunc) {{.cacheSharedIndexInformer|raw}}
}
`
