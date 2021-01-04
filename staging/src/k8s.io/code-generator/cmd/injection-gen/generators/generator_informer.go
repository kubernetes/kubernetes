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

	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
	"k8s.io/klog/v2"
)

// injectionTestGenerator produces a file of listers for a given GroupVersion and
// type.
type injectionGenerator struct {
	generator.DefaultGen
	outputPackage               string
	groupVersion                clientgentypes.GroupVersion
	groupGoName                 string
	typeToGenerate              *types.Type
	imports                     namer.ImportTracker
	typedInformerPackage        string
	groupInformerFactoryPackage string
}

var _ generator.Generator = (*injectionGenerator)(nil)

func (g *injectionGenerator) Filter(c *generator.Context, t *types.Type) bool {
	// Only process the type for this informer generator.
	return t == g.typeToGenerate
}

func (g *injectionGenerator) Namers(c *generator.Context) namer.NameSystems {
	publicPluralNamer := &ExceptionNamer{
		Exceptions: map[string]string{
			// these exceptions are used to deconflict the generated code
			// you can put your fully qualified package like
			// to generate a name that doesn't conflict with your group.
			// "k8s.io/apis/events/v1beta1.Event": "EventResource"
		},
		KeyFunc: func(t *types.Type) string {
			return t.Name.Package + "." + t.Name.Name
		},
		Delegate: namer.NewPublicPluralNamer(map[string]string{
			"Endpoints": "Endpoints",
		}),
	}

	return namer.NameSystems{
		"raw":          namer.NewRawNamer(g.outputPackage, g.imports),
		"publicPlural": publicPluralNamer,
	}
}

func (g *injectionGenerator) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	return
}

func (g *injectionGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "{{", "}}")

	klog.V(5).Info("processing type ", t)

	m := map[string]interface{}{
		"group":                     namer.IC(g.groupGoName),
		"type":                      t,
		"version":                   namer.IC(g.groupVersion.Version.String()),
		"injectionRegisterInformer": c.Universe.Function(types.Name{Package: "k8s.io/client-go/injection", Name: "Default.RegisterInformer"}),
		"controllerInformer":        c.Universe.Type(types.Name{Package: "k8s.io/client-go/injection", Name: "Informer"}),
		"informersTypedInformer":    c.Universe.Type(types.Name{Package: g.typedInformerPackage, Name: t.Name.Name + "Informer"}),
		"factoryGet":                c.Universe.Type(types.Name{Package: g.groupInformerFactoryPackage, Name: "Get"}),
		"klogFatal":                 c.Universe.Function(types.Name{Package: "k8s.io/klog/v2", Name: "Fatal"}),
		"contextContext":            c.Universe.Type(types.Name{Package: "context", Name: "Context"}),
	}

	sw.Do(injectionInformer, m)

	return sw.Error()
}

var injectionInformer = `
func init() {
	{{.injectionRegisterInformer|raw}}(withInformer)
}

// Key is used for associating the Informer inside the context.Context.
type Key struct{}

func withInformer(ctx {{.contextContext|raw}}) ({{.contextContext|raw}}, {{.controllerInformer|raw}}) {
	f := {{.factoryGet|raw}}(ctx)
	inf := f.{{.group}}().{{.version}}().{{.type|publicPlural}}()
	return context.WithValue(ctx, Key{}, inf), inf.Informer()
}

// Get extracts the typed informer from the context.
func Get(ctx {{.contextContext|raw}}) {{.informersTypedInformer|raw}} {
	untyped := ctx.Value(Key{})
	if untyped == nil {
		{{.klogFatal|raw}}("Unable to fetch {{.informersTypedInformer|raw}} from context.")
	}
	return untyped.({{.informersTypedInformer|raw}})
}
`
