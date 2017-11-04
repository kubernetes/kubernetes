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

	"k8s.io/code-generator/cmd/client-gen/generators/util"
	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"

	"github.com/golang/glog"
)

// informerGenerator produces a file of listers for a given GroupVersion and
// type.
type informerGenerator struct {
	generator.DefaultGen
	outputPackage             string
	groupVersion              clientgentypes.GroupVersion
	typeToGenerate            *types.Type
	imports                   namer.ImportTracker
	clientSetPackage          string
	listersPackage            string
	internalInterfacesPackage string
}

var _ generator.Generator = &informerGenerator{}

func (g *informerGenerator) Filter(c *generator.Context, t *types.Type) bool {
	return t == g.typeToGenerate
}

func (g *informerGenerator) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *informerGenerator) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	return
}

func (g *informerGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	glog.V(5).Infof("processing type %v", t)

	//listerPackage := "k8s.io/kubernetes/pkg/client/listers/" + g.groupVersion.Group.NonEmpty() + "/" + strings.ToLower(g.groupVersion.Version.NonEmpty())
	listerPackage := fmt.Sprintf("%s/%s/%s", g.listersPackage, g.groupVersion.Group.NonEmpty(), strings.ToLower(g.groupVersion.Version.NonEmpty()))
	clientSetInterface := c.Universe.Type(types.Name{Package: g.clientSetPackage, Name: "Interface"})
	informerFor := "InformerFor"

	tags, err := util.ParseClientGenTags(t.SecondClosestCommentLines)
	if err != nil {
		return err
	}

	m := map[string]interface{}{
		"apiScheme":                       c.Universe.Type(apiScheme),
		"cacheIndexers":                   c.Universe.Type(cacheIndexers),
		"cacheListWatch":                  c.Universe.Type(cacheListWatch),
		"cacheMetaNamespaceIndexFunc":     c.Universe.Function(cacheMetaNamespaceIndexFunc),
		"cacheNamespaceIndex":             c.Universe.Variable(cacheNamespaceIndex),
		"cacheNewSharedIndexInformer":     c.Universe.Function(cacheNewSharedIndexInformer),
		"cacheSharedIndexInformer":        c.Universe.Type(cacheSharedIndexInformer),
		"clientSetInterface":              clientSetInterface,
		"group":                           namer.IC(g.groupVersion.Group.NonEmpty()),
		"informerFor":                     informerFor,
		"interfacesSharedInformerFactory": c.Universe.Type(types.Name{Package: g.internalInterfacesPackage, Name: "SharedInformerFactory"}),
		"listOptions":                     c.Universe.Type(listOptions),
		"lister":                          c.Universe.Type(types.Name{Package: listerPackage, Name: t.Name.Name + "Lister"}),
		"namespaceAll":                    c.Universe.Type(metav1NamespaceAll),
		"namespaced":                      !tags.NonNamespaced,
		"newLister":                       c.Universe.Function(types.Name{Package: listerPackage, Name: "New" + t.Name.Name + "Lister"}),
		"runtimeObject":                   c.Universe.Type(runtimeObject),
		"timeDuration":                    c.Universe.Type(timeDuration),
		"type":                            t,
		"v1ListOptions":                   c.Universe.Type(v1ListOptions),
		"version":                         namer.IC(g.groupVersion.Version.String()),
		"watchInterface":                  c.Universe.Type(watchInterface),
	}

	sw.Do(typeInformerInterface, m)
	sw.Do(typeInformerStruct, m)
	sw.Do(typeInformerPublicConstructor, m)
	sw.Do(typeInformerConstructor, m)
	sw.Do(typeInformerInformer, m)
	sw.Do(typeInformerLister, m)

	return sw.Error()
}

var typeInformerInterface = `
// $.type|public$Informer provides access to a shared informer and lister for
// $.type|publicPlural$.
type $.type|public$Informer interface {
	Informer() $.cacheSharedIndexInformer|raw$
	Lister() $.lister|raw$
}
`

var typeInformerStruct = `
type $.type|private$Informer struct {
	factory $.interfacesSharedInformerFactory|raw$
}
`

var typeInformerPublicConstructor = `
// New$.type|public$Informer constructs a new informer for $.type|public$ type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func New$.type|public$Informer(client $.clientSetInterface|raw$$if .namespaced$, namespace string$end$, resyncPeriod $.timeDuration|raw$, indexers $.cacheIndexers|raw$) $.cacheSharedIndexInformer|raw$ {
	return $.cacheNewSharedIndexInformer|raw$(
		&$.cacheListWatch|raw${
			ListFunc: func(options $.v1ListOptions|raw$) ($.runtimeObject|raw$, error) {
				return client.$.group$$.version$().$.type|publicPlural$($if .namespaced$namespace$end$).List(options)
			},
			WatchFunc: func(options $.v1ListOptions|raw$) ($.watchInterface|raw$, error) {
				return client.$.group$$.version$().$.type|publicPlural$($if .namespaced$namespace$end$).Watch(options)
			},
		},
		&$.type|raw${},
		resyncPeriod,
		indexers,
	)
}
`

var typeInformerConstructor = `
func default$.type|public$Informer(client $.clientSetInterface|raw$, resyncPeriod $.timeDuration|raw$) $.cacheSharedIndexInformer|raw$ {
	return New$.type|public$Informer(client, $if .namespaced$$.namespaceAll|raw$, $end$resyncPeriod, $.cacheIndexers|raw${$.cacheNamespaceIndex|raw$: $.cacheMetaNamespaceIndexFunc|raw$})
}
`

var typeInformerInformer = `
func (f *$.type|private$Informer) Informer() $.cacheSharedIndexInformer|raw$ {
	return f.factory.$.informerFor$(&$.type|raw${}, default$.type|public$Informer)
}
`

var typeInformerLister = `
func (f *$.type|private$Informer) Lister() $.lister|raw$ {
	return $.newLister|raw$(f.Informer().GetIndexer())
}
`
