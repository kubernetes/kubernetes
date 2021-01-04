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

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"

	"k8s.io/klog/v2"
)

// factoryTestGenerator produces a file of factory injection of a given type.
type factoryGenerator struct {
	generator.DefaultGen
	outputPackage                string
	imports                      namer.ImportTracker
	cachingClientSetPackage      string
	sharedInformerFactoryPackage string
	filtered                     bool
}

var _ generator.Generator = (*factoryGenerator)(nil)

func (g *factoryGenerator) Filter(c *generator.Context, t *types.Type) bool {
	// We generate a single factory, so return true once.
	if !g.filtered {
		g.filtered = true
		return true
	}
	return false
}

func (g *factoryGenerator) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *factoryGenerator) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	return
}

func (g *factoryGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "{{", "}}")

	klog.V(5).Info("processing type ", t)

	m := map[string]interface{}{
		"cachingClientGet": c.Universe.Type(types.Name{Package: g.cachingClientSetPackage, Name: "Get"}),
		"informersNewSharedInformerFactoryWithOptions": c.Universe.Function(types.Name{Package: g.sharedInformerFactoryPackage, Name: "NewSharedInformerFactoryWithOptions"}),
		"informersSharedInformerOption":                c.Universe.Function(types.Name{Package: g.sharedInformerFactoryPackage, Name: "SharedInformerOption"}),
		"informersWithNamespace":                       c.Universe.Function(types.Name{Package: g.sharedInformerFactoryPackage, Name: "WithNamespace"}),
		"informersSharedInformerFactory":               c.Universe.Function(types.Name{Package: g.sharedInformerFactoryPackage, Name: "SharedInformerFactory"}),
		"injectionRegisterInformerFactory":             c.Universe.Function(types.Name{Package: "k8s.io/client-go/injection", Name: "Default.RegisterInformerFactory"}),
		"injectionHasNamespace":                        c.Universe.Function(types.Name{Package: "k8s.io/client-go/injection", Name: "HasNamespaceScope"}),
		"injectionGetNamespace":                        c.Universe.Function(types.Name{Package: "k8s.io/client-go/injection", Name: "GetNamespaceScope"}),
		"controllerGetResyncPeriod":                    c.Universe.Function(types.Name{Package: "k8s.io/client-go/controller", Name: "GetResyncPeriod"}),
		"klogFatal":                                    c.Universe.Function(types.Name{Package: "k8s.io/klog/v2", Name: "Fatal"}),
		"contextContext":                               c.Universe.Type(types.Name{Package: "context", Name: "Context"}),
	}

	sw.Do(injectionFactory, m)

	return sw.Error()
}

var injectionFactory = `
func init() {
	{{.injectionRegisterInformerFactory|raw}}(withInformerFactory)
}

// Key is used as the key for associating information with a context.Context.
type Key struct{}

func withInformerFactory(ctx {{.contextContext|raw}}) {{.contextContext|raw}} {
	c := {{.cachingClientGet|raw}}(ctx)
	opts := make([]{{.informersSharedInformerOption|raw}}, 0, 1)
	if {{.injectionHasNamespace|raw}}(ctx) {
		opts = append(opts, {{.informersWithNamespace|raw}}({{.injectionGetNamespace|raw}}(ctx)))
	}
	return context.WithValue(ctx, Key{},
		{{.informersNewSharedInformerFactoryWithOptions|raw}}(c, {{.controllerGetResyncPeriod|raw}}(ctx), opts...))
}

// Get extracts the InformerFactory from the context.
func Get(ctx {{.contextContext|raw}}) {{.informersSharedInformerFactory|raw}} {
	untyped := ctx.Value(Key{})
	if untyped == nil {
		{{.klogFatal|raw}}("Unable to fetch {{.informersSharedInformerFactory}} from context.")
	}
	return untyped.({{.informersSharedInformerFactory|raw}})
}
`
