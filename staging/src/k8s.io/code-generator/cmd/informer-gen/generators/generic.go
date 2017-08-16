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
	"sort"
	"strings"

	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

// genericGenerator generates the generic informer.
type genericGenerator struct {
	generator.DefaultGen
	outputPackage        string
	imports              namer.ImportTracker
	groupVersions        map[string]clientgentypes.GroupVersions
	typesForGroupVersion map[clientgentypes.GroupVersion][]*types.Type
	filtered             bool
}

var _ generator.Generator = &genericGenerator{}

func (g *genericGenerator) Filter(c *generator.Context, t *types.Type) bool {
	if !g.filtered {
		g.filtered = true
		return true
	}
	return false
}

func (g *genericGenerator) Namers(c *generator.Context) namer.NameSystems {
	pluralExceptions := map[string]string{
		"Endpoints": "Endpoints",
	}
	return namer.NameSystems{
		"raw":                namer.NewRawNamer(g.outputPackage, g.imports),
		"allLowercasePlural": namer.NewAllLowercasePluralNamer(pluralExceptions),
		"publicPlural":       namer.NewPublicPluralNamer(pluralExceptions),
	}
}

func (g *genericGenerator) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	imports = append(imports, "fmt")
	return
}

type group struct {
	Name     string
	Versions []*version
}

type groupSort []group

func (g groupSort) Len() int           { return len(g) }
func (g groupSort) Less(i, j int) bool { return strings.ToLower(g[i].Name) < strings.ToLower(g[j].Name) }
func (g groupSort) Swap(i, j int)      { g[i], g[j] = g[j], g[i] }

type version struct {
	Name      string
	Resources []*types.Type
}

type versionSort []*version

func (v versionSort) Len() int { return len(v) }
func (v versionSort) Less(i, j int) bool {
	return strings.ToLower(v[i].Name) < strings.ToLower(v[j].Name)
}
func (v versionSort) Swap(i, j int) { v[i], v[j] = v[j], v[i] }

func (g *genericGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "{{", "}}")

	groups := []group{}
	schemeGVs := make(map[*version]*types.Type)

	orderer := namer.Orderer{Namer: namer.NewPrivateNamer(0)}
	for _, groupVersions := range g.groupVersions {
		group := group{
			Name:     namer.IC(groupVersions.Group.NonEmpty()),
			Versions: []*version{},
		}
		for _, v := range groupVersions.Versions {
			gv := clientgentypes.GroupVersion{Group: groupVersions.Group, Version: v}
			version := &version{
				Name:      namer.IC(v.NonEmpty()),
				Resources: orderer.OrderTypes(g.typesForGroupVersion[gv]),
			}
			schemeGVs[version] = c.Universe.Variable(types.Name{Package: g.typesForGroupVersion[gv][0].Name.Package, Name: "SchemeGroupVersion"})
			group.Versions = append(group.Versions, version)
		}
		sort.Sort(versionSort(group.Versions))
		groups = append(groups, group)
	}
	sort.Sort(groupSort(groups))

	m := map[string]interface{}{
		"cacheGenericLister":         c.Universe.Type(cacheGenericLister),
		"cacheNewGenericLister":      c.Universe.Function(cacheNewGenericLister),
		"cacheSharedIndexInformer":   c.Universe.Type(cacheSharedIndexInformer),
		"groups":                     groups,
		"schemeGVs":                  schemeGVs,
		"schemaGroupResource":        c.Universe.Type(schemaGroupResource),
		"schemaGroupVersionResource": c.Universe.Type(schemaGroupVersionResource),
	}

	sw.Do(genericInformer, m)
	sw.Do(forResource, m)

	return sw.Error()
}

var genericInformer = `
// GenericInformer is type of SharedIndexInformer which will locate and delegate to other
// sharedInformers based on type
type GenericInformer interface {
	Informer() {{.cacheSharedIndexInformer|raw}}
	Lister() {{.cacheGenericLister|raw}}
}

type genericInformer struct {
	informer {{.cacheSharedIndexInformer|raw}}
	resource {{.schemaGroupResource|raw}}
}

// Informer returns the SharedIndexInformer.
func (f *genericInformer) Informer() {{.cacheSharedIndexInformer|raw}} {
	return f.informer
}

// Lister returns the GenericLister.
func (f *genericInformer) Lister() {{.cacheGenericLister|raw}} {
	return {{.cacheNewGenericLister|raw}}(f.Informer().GetIndexer(), f.resource)
}
`

var forResource = `
// ForResource gives generic access to a shared informer of the matching type
// TODO extend this to unknown resources with a client pool
func (f *sharedInformerFactory) ForResource(resource {{.schemaGroupVersionResource|raw}}) (GenericInformer, error) {
	switch resource {
		{{range $group := .groups -}}
			{{range $version := .Versions -}}
		// Group={{$group.Name}}, Version={{.Name}}
				{{range .Resources -}}
	case {{index $.schemeGVs $version|raw}}.WithResource("{{.|allLowercasePlural}}"):
		return &genericInformer{resource: resource.GroupResource(), informer: f.{{$group.Name}}().{{$version.Name}}().{{.|publicPlural}}().Informer()}, nil
				{{end}}
			{{end}}
		{{end -}}
	}

	return nil, fmt.Errorf("no informer found for %v", resource)
}
`
