/*
Copyright 2020 The Kubernetes Authors.

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

// fakeListerGenerator produces a file of listers for a given GroupVersion and
// type.
type fakeListerGenerator struct {
	generator.DefaultGen
	outputPackage  string
	typeToGenerate *types.Type
	imports        namer.ImportTracker
	objectMeta     *types.Type
	listersPackage string
}

var _ generator.Generator = &fakeListerGenerator{}

func (g *fakeListerGenerator) Filter(c *generator.Context, t *types.Type) bool {
	return t == g.typeToGenerate
}

func (g *fakeListerGenerator) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *fakeListerGenerator) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	imports = append(imports, "k8s.io/apimachinery/pkg/api/errors")
	imports = append(imports, "k8s.io/apimachinery/pkg/labels")
	return
}

func (g *fakeListerGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	klog.V(5).Infof("processing type %v", t)
	m := map[string]interface{}{
		"Resource":              c.Universe.Function(types.Name{Package: t.Name.Package, Name: "Resource"}),
		"type":                  t,
		"objectMeta":            g.objectMeta,
		"realListerConstructor": c.Universe.Function(types.Name{Package: g.listersPackage, Name: "New" + t.Name.Name + "Lister"}),
		"realListerInterface":   c.Universe.Type(types.Name{Package: g.listersPackage, Name: t.Name.Name + "Lister"}),
		"newDefaultIndexer":     c.Universe.Function(types.Name{Package: g.listersPackage, Name: "New" + t.Name.Name + "DefaultIndexer"}),
		"newCacheIndexer":       c.Universe.Function(cacheNewIndexer),
		"cacheKeyFunc":          c.Universe.Function(cacheKeyFunc),
		"cacheIndexer":          c.Universe.Type(cacheIndexer),
		"listerInstanceName":    t.Name.Name + "Lister",
	}

	sw.Do(fakeTypeListerInterface, m)
	sw.Do(fakeTypeListerStruct, m)
	sw.Do(fakeTypeListerConstructor, m)

	sw.Do(fakeTypeLister_Add, m)
	sw.Do(fakeTypeLister_Update, m)
	sw.Do(fakeTypeLister_Delete, m)

	return sw.Error()
}

var fakeTypeListerInterface = `
// $.type|public$Lister helps list $.type|publicPlural$.
// All objects returned here must be treated as read-only.
type Fake$.type|public$Lister interface {
	$.realListerInterface|raw$
	// Add adds the given object to the lister
	Add(obj ...*$.type|raw$) error
	// Update updates the given object in the lister
	Update(obj *$.type|raw$) error
	// Delete deletes the given object from lister
	Delete(obj *$.type|raw$) error
}
`

var fakeTypeListerStruct = `
// $.type|private$Lister implements the $.type|public$Lister interface.
type $.type|private$Lister struct {
	index $.cacheIndexer|raw$
	$.realListerInterface|raw$
}
`

var fakeTypeListerConstructor = `
// New$.type|public$Lister returns a new $.type|public$Lister.
func NewFake$.type|public$Lister() Fake$.type|public$Lister {
	indexers := $.newDefaultIndexer|raw$()
	index := $.newCacheIndexer|raw$($.cacheKeyFunc|raw$, indexers)
	lister := $.realListerConstructor|raw$(index)
	return &$.type|private$Lister{
		index: index,
		$.listerInstanceName$: lister,
	}
}
`

var fakeTypeLister_Add = `
// Add adds the given object to the lister
func (s *$.type|private$Lister) Add(obj ...*$.type|raw$) error {
	for _, curr := range obj{
		if err := s.index.Add(curr);err != nil{
			return err
		}
	}
	return nil
}
`

var fakeTypeLister_Update = `
// Update updates the given object in the lister
func (s *$.type|private$Lister) Update(obj *$.type|raw$) error {
	return s.index.Update(obj)
}
`

var fakeTypeLister_Delete = `
// Delete deletes the given object from lister
func (s *$.type|private$Lister) Delete(obj *$.type|raw$) error {
	return s.index.Delete(obj)
}
`
