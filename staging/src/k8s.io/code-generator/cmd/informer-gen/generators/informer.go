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

	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"

	"k8s.io/code-generator/cmd/client-gen/generators/util"
	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"

	"k8s.io/klog/v2"
)

// informerGenerator produces a file of listers for a given GroupVersion and
// type.
type informerGenerator struct {
	generator.GoGenerator
	outputPackage             string
	groupPkgName              string
	groupVersion              clientgentypes.GroupVersion
	groupGoName               string
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

	klog.V(5).Infof("processing type %v", t)

	listerPackage := fmt.Sprintf("%s/%s/%s", g.listersPackage, g.groupPkgName, strings.ToLower(g.groupVersion.Version.NonEmpty()))
	clientSetInterface := c.Universe.Type(types.Name{Package: g.clientSetPackage, Name: "Interface"})
	informerFor := "InformerFor"

	tags, err := util.ParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
	if err != nil {
		return err
	}

	m := map[string]interface{}{
		"apiScheme":                                c.Universe.Type(apiScheme),
		"cacheIndexers":                            c.Universe.Type(cacheIndexers),
		"cacheListWatch":                           c.Universe.Type(cacheListWatch),
		"cacheMetaNamespaceIndexFunc":              c.Universe.Function(cacheMetaNamespaceIndexFunc),
		"cacheNamespaceIndex":                      c.Universe.Variable(cacheNamespaceIndex),
		"cacheNewSharedIndexInformer":              c.Universe.Function(cacheNewSharedIndexInformer),
		"cacheNewSharedIndexInformerWithOptions":   c.Universe.Function(cacheNewSharedIndexInformerWithOptions),
		"cacheSharedIndexInformer":                 c.Universe.Type(cacheSharedIndexInformer),
		"cacheSharedIndexInformerOptions":          c.Universe.Type(cacheSharedIndexInformerOptions),
		"cacheToListWatcherWithWatchListSemantics": c.Universe.Function(cacheToListWatcherWithWatchListSemanticsFunc),
		"cacheInformerName":                        c.Universe.Type(cacheInformerName),
		"clientSetInterface":                       clientSetInterface,
		"contextContext":                           c.Universe.Type(contextContext),
		"contextBackground":                        c.Universe.Function(contextBackgroundFunc),
		"group":                                    namer.IC(g.groupGoName),
		"groupName":                                g.groupVersion.Group.String(),
		"informerFor":                              informerFor,
		"interfacesInformerOptions":                c.Universe.Type(types.Name{Package: g.internalInterfacesPackage, Name: "InformerOptions"}),
		"interfacesTweakListOptionsFunc":           c.Universe.Type(types.Name{Package: g.internalInterfacesPackage, Name: "TweakListOptionsFunc"}),
		"interfacesSharedInformerFactory":          c.Universe.Type(types.Name{Package: g.internalInterfacesPackage, Name: "SharedInformerFactory"}),
		"listOptions":                              c.Universe.Type(listOptions),
		"lister":                                   c.Universe.Type(types.Name{Package: listerPackage, Name: t.Name.Name + "Lister"}),
		"namespaceAll":                             c.Universe.Type(metav1NamespaceAll),
		"namespaced":                               !tags.NonNamespaced,
		"newLister":                                c.Universe.Function(types.Name{Package: listerPackage, Name: "New" + t.Name.Name + "Lister"}),
		"resourceName":                             strings.ToLower(t.Name.Name) + "s",
		"runtimeObject":                            c.Universe.Type(runtimeObject),
		"schemaGroupVersionResource":               c.Universe.Type(schemaGroupVersionResource),
		"timeDuration":                             c.Universe.Type(timeDuration),
		"type":                                     t,
		"v1ListOptions":                            c.Universe.Type(v1ListOptions),
		"version":                                  namer.IC(g.groupVersion.Version.String()),
		"versionName":                              g.groupVersion.Version.String(),
		"watchInterface":                           c.Universe.Type(watchInterface),
	}

	sw.Do(typeInformerInterface, m)
	sw.Do(typeInformerStruct, m)
	sw.Do(typeInformerPublicConstructor, m)
	sw.Do(typeFilteredInformerPublicConstructor, m)
	sw.Do(typeInformerPublicConstructorWithOptions, m)
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
	tweakListOptions $.interfacesTweakListOptionsFunc|raw$
	$if .namespaced$namespace string$end$
}
`

var typeInformerPublicConstructor = `
// New$.type|public$Informer constructs a new informer for $.type|public$ type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func New$.type|public$Informer(client $.clientSetInterface|raw$$if .namespaced$, namespace string$end$, resyncPeriod $.timeDuration|raw$, indexers $.cacheIndexers|raw$) $.cacheSharedIndexInformer|raw$ {
	return New$.type|public$InformerWithOptions(client$if .namespaced$, namespace$end$, $.interfacesInformerOptions|raw${ResyncPeriod: resyncPeriod, Indexers: indexers})
}
`

var typeFilteredInformerPublicConstructor = `
// NewFiltered$.type|public$Informer constructs a new informer for $.type|public$ type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func NewFiltered$.type|public$Informer(client $.clientSetInterface|raw$$if .namespaced$, namespace string$end$, resyncPeriod $.timeDuration|raw$, indexers $.cacheIndexers|raw$, tweakListOptions $.interfacesTweakListOptionsFunc|raw$) $.cacheSharedIndexInformer|raw$ {
	return New$.type|public$InformerWithOptions(client$if .namespaced$, namespace$end$, $.interfacesInformerOptions|raw${ResyncPeriod: resyncPeriod, Indexers: indexers, TweakListOptions: tweakListOptions})
}
`

var typeInformerPublicConstructorWithOptions = `
// New$.type|public$InformerWithOptions constructs a new informer for $.type|public$ type with additional options.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func New$.type|public$InformerWithOptions(client $.clientSetInterface|raw$$if .namespaced$, namespace string$end$, options $.interfacesInformerOptions|raw$) $.cacheSharedIndexInformer|raw$ {
	gvr := $.schemaGroupVersionResource|raw${Group: "$.groupName$", Version: "$.versionName$", Resource: "$.resourceName$"}
	identifier := options.InformerName.WithResource(gvr)
	tweakListOptions := options.TweakListOptions
	return $.cacheNewSharedIndexInformerWithOptions|raw$(
		$.cacheToListWatcherWithWatchListSemantics|raw$(&$.cacheListWatch|raw${
			ListFunc: func(opts $.v1ListOptions|raw$) ($.runtimeObject|raw$, error) {
				if tweakListOptions != nil {
					tweakListOptions(&opts)
				}
				return client.$.group$$.version$().$.type|publicPlural$($if .namespaced$namespace$end$).List($.contextBackground|raw$(), opts)
			},
			WatchFunc: func(opts $.v1ListOptions|raw$) ($.watchInterface|raw$, error) {
				if tweakListOptions != nil {
					tweakListOptions(&opts)
				}
				return client.$.group$$.version$().$.type|publicPlural$($if .namespaced$namespace$end$).Watch($.contextBackground|raw$(), opts)
			},
			ListWithContextFunc: func(ctx $.contextContext|raw$, opts $.v1ListOptions|raw$) ($.runtimeObject|raw$, error) {
				if tweakListOptions != nil {
					tweakListOptions(&opts)
				}
				return client.$.group$$.version$().$.type|publicPlural$($if .namespaced$namespace$end$).List(ctx, opts)
			},
			WatchFuncWithContext: func(ctx $.contextContext|raw$, opts $.v1ListOptions|raw$) ($.watchInterface|raw$, error) {
				if tweakListOptions != nil {
					tweakListOptions(&opts)
				}
				return client.$.group$$.version$().$.type|publicPlural$($if .namespaced$namespace$end$).Watch(ctx, opts)
			},
		}, client),
		&$.type|raw${},
		$.cacheSharedIndexInformerOptions|raw${
			ResyncPeriod: options.ResyncPeriod,
			Indexers:     options.Indexers,
			Identifier:   identifier,
		},
	)
}
`

var typeInformerConstructor = `
func (f *$.type|private$Informer) defaultInformer(client $.clientSetInterface|raw$, resyncPeriod $.timeDuration|raw$) $.cacheSharedIndexInformer|raw$ {
	return New$.type|public$InformerWithOptions(client$if .namespaced$, f.namespace$end$, $.interfacesInformerOptions|raw${ResyncPeriod: resyncPeriod, Indexers: $.cacheIndexers|raw${$.cacheNamespaceIndex|raw$: $.cacheMetaNamespaceIndexFunc|raw$}, InformerName: f.factory.InformerName(), TweakListOptions: f.tweakListOptions})
}
`

var typeInformerInformer = `
func (f *$.type|private$Informer) Informer() $.cacheSharedIndexInformer|raw$ {
	return f.factory.$.informerFor$(&$.type|raw${}, f.defaultInformer)
}
`

var typeInformerLister = `
func (f *$.type|private$Informer) Lister() $.lister|raw$ {
	return $.newLister|raw$(f.Informer().GetIndexer())
}
`
