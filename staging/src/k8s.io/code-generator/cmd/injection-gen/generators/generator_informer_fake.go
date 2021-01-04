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

// fakeInformerGenerator produces a file of listers for a given GroupVersion and
// type.
type fakeInformerGenerator struct {
	generator.DefaultGen
	outputPackage string
	imports       namer.ImportTracker

	typeToGenerate          *types.Type
	groupVersion            clientgentypes.GroupVersion
	groupGoName             string
	informerInjectionPkg    string
	fakeFactoryInjectionPkg string
}

var _ generator.Generator = (*fakeInformerGenerator)(nil)

func (g *fakeInformerGenerator) Filter(c *generator.Context, t *types.Type) bool {
	// Only process the type for this informer generator.
	return t == g.typeToGenerate
}

func (g *fakeInformerGenerator) Namers(c *generator.Context) namer.NameSystems {
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

func (g *fakeInformerGenerator) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	return
}

func (g *fakeInformerGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "{{", "}}")

	klog.V(5).Info("processing type ", t)

	m := map[string]interface{}{
		"informerKey":               c.Universe.Type(types.Name{Package: g.informerInjectionPkg, Name: "Key"}),
		"informerGet":               c.Universe.Function(types.Name{Package: g.informerInjectionPkg, Name: "Get"}),
		"factoryGet":                c.Universe.Function(types.Name{Package: g.fakeFactoryInjectionPkg, Name: "Get"}),
		"group":                     namer.IC(g.groupGoName),
		"type":                      t,
		"version":                   namer.IC(g.groupVersion.Version.String()),
		"controllerInformer":        c.Universe.Type(types.Name{Package: "k8s.io/client-go//controller", Name: "Informer"}),
		"injectionRegisterInformer": c.Universe.Function(types.Name{Package: "k8s.io/client-go/injection", Name: "Fake.RegisterInformer"}),
		"contextContext":            c.Universe.Type(types.Name{Package: "context", Name: "Context"}),
	}

	sw.Do(injectionFakeInformer, m)

	return sw.Error()
}

var injectionFakeInformer = `
var Get = {{.informerGet|raw}}

func init() {
	{{.injectionRegisterInformer|raw}}(withInformer)
}

func withInformer(ctx {{.contextContext|raw}}) ({{.contextContext|raw}}, {{.controllerInformer|raw}}) {
	f := {{.factoryGet|raw}}(ctx)
	inf := f.{{.group}}().{{.version}}().{{.type|publicPlural}}()
	return context.WithValue(ctx, {{.informerKey|raw}}{}, inf), inf.Informer()
}
`
