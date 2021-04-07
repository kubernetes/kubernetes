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
	"sort"
	"strings"

	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

// utilGenerator generates the ForKind() utility function.
type utilGenerator struct {
	generator.DefaultGen
	outputPackage        string
	imports              namer.ImportTracker
	groupVersions        map[string]clientgentypes.GroupVersions
	groupGoNames         map[string]string
	typesForGroupVersion map[clientgentypes.GroupVersion][]applyConfig
	filtered             bool
}

var _ generator.Generator = &utilGenerator{}

func (g *utilGenerator) Filter(*generator.Context, *types.Type) bool {
	// generate file exactly once
	if !g.filtered {
		g.filtered = true
		return true
	}
	return false
}

func (g *utilGenerator) Namers(*generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw":          namer.NewRawNamer(g.outputPackage, g.imports),
		"singularKind": namer.NewPublicNamer(0),
	}
}

func (g *utilGenerator) Imports(*generator.Context) (imports []string) {
	return g.imports.ImportLines()
}

type group struct {
	GroupGoName string
	Name        string
	Versions    []*version
}

type groupSort []group

func (g groupSort) Len() int { return len(g) }
func (g groupSort) Less(i, j int) bool {
	return strings.ToLower(g[i].Name) < strings.ToLower(g[j].Name)
}
func (g groupSort) Swap(i, j int) { g[i], g[j] = g[j], g[i] }

type version struct {
	Name      string
	GoName    string
	Resources []applyConfig
}

type versionSort []*version

func (v versionSort) Len() int { return len(v) }
func (v versionSort) Less(i, j int) bool {
	return strings.ToLower(v[i].Name) < strings.ToLower(v[j].Name)
}
func (v versionSort) Swap(i, j int) { v[i], v[j] = v[j], v[i] }

type applyConfig struct {
	Type               *types.Type
	ApplyConfiguration *types.Type
}

type applyConfigSort []applyConfig

func (v applyConfigSort) Len() int { return len(v) }
func (v applyConfigSort) Less(i, j int) bool {
	return strings.ToLower(v[i].Type.Name.Name) < strings.ToLower(v[j].Type.Name.Name)
}
func (v applyConfigSort) Swap(i, j int) { v[i], v[j] = v[j], v[i] }

func (g *utilGenerator) GenerateType(c *generator.Context, _ *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "{{", "}}")

	var groups []group
	schemeGVs := make(map[*version]*types.Type)

	for groupPackageName, groupVersions := range g.groupVersions {
		group := group{
			GroupGoName: g.groupGoNames[groupPackageName],
			Name:        groupVersions.Group.NonEmpty(),
			Versions:    []*version{},
		}
		for _, v := range groupVersions.Versions {
			gv := clientgentypes.GroupVersion{Group: groupVersions.Group, Version: v.Version}
			version := &version{
				Name:      v.Version.NonEmpty(),
				GoName:    namer.IC(v.Version.NonEmpty()),
				Resources: g.typesForGroupVersion[gv],
			}
			schemeGVs[version] = c.Universe.Variable(types.Name{
				Package: g.typesForGroupVersion[gv][0].Type.Name.Package,
				Name:    "SchemeGroupVersion",
			})
			group.Versions = append(group.Versions, version)
		}
		sort.Sort(versionSort(group.Versions))
		groups = append(groups, group)
	}
	sort.Sort(groupSort(groups))

	m := map[string]interface{}{
		"groups":                 groups,
		"schemeGVs":              schemeGVs,
		"schemaGroupVersionKind": groupVersionKind,
		"applyConfiguration":     applyConfiguration,
	}
	sw.Do(forKindFunc, m)

	return sw.Error()
}

var forKindFunc = `
// ForKind returns an apply configuration type for the given GroupVersionKind, or nil if no
// apply configuration type exists for the given GroupVersionKind.
func ForKind(kind {{.schemaGroupVersionKind|raw}}) interface{} {
	switch kind {
		{{range $group := .groups -}}{{$GroupGoName := .GroupGoName -}}
			{{range $version := .Versions -}}
	// Group={{$group.Name}}, Version={{.Name}}
				{{range .Resources -}}
	case {{index $.schemeGVs $version|raw}}.WithKind("{{.Type|singularKind}}"):
		return &{{.ApplyConfiguration|raw}}{}
				{{end}}
			{{end}}
		{{end -}}
	}
	return nil
}
`
