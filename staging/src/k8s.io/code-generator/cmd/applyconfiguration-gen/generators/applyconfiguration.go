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

	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
)

// applyConfigurationGenerator produces apply configurations for a given GroupVersion and type.
type applyConfigurationGenerator struct {
	generator.DefaultGen
	outputPackage string
	localPackage  types.Name
	groupVersion  clientgentypes.GroupVersion
	applyConfig   applyConfig
	imports       namer.ImportTracker
	refGraph      refGraph
}

var _ generator.Generator = &applyConfigurationGenerator{}

func (g *applyConfigurationGenerator) Filter(_ *generator.Context, t *types.Type) bool {
	return t == g.applyConfig.Type
}

func (g *applyConfigurationGenerator) Namers(*generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.localPackage.Package, g.imports),
	}
}

func (g *applyConfigurationGenerator) Imports(*generator.Context) (imports []string) {
	return g.imports.ImportLines()
}

// TypeParams provides a struct that an apply configuration
// is generated for as well as the apply configuration details
// and types referenced by the struct.
type TypeParams struct {
	Struct      *types.Type
	ApplyConfig applyConfig
	Refs        map[string]*types.Type
}

type memberParams struct {
	TypeParams
	Member     types.Member
	MemberType *types.Type
	JSONTags   JSONTags
}

func (g *applyConfigurationGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	klog.V(5).Infof("processing type %v", t)
	typeParams := TypeParams{
		Struct:      t,
		ApplyConfig: g.applyConfig,
		Refs: map[string]*types.Type{
			"unstructuredConverter": unstructuredConverter,
			"unstructured":          unstructured,
			"jsonUnmarshal":         jsonUnmarshal,
			"jsonMarshal":           jsonMarshal,
		},
	}

	g.generateStruct(t, sw, typeParams)

	for _, member := range t.Members {
		memberType := g.refGraph.applyConfigForType(member.Type)
		if _, ok := lookupJSONTags(member); ok {
			if g.refGraph.isApplyConfig(member.Type) {
				memberType = &types.Type{Kind: types.Pointer, Elem: memberType}
			}
		}
	}

	sw.Do(listAlias, typeParams)
	sw.Do(mapAlias, typeParams)

	return sw.Error()
}

func (g *applyConfigurationGenerator) generateStruct(t *types.Type, sw *generator.SnippetWriter, typeParams TypeParams) {
	sw.Do("// $.ApplyConfig.ApplyConfiguration|public$ represents an declarative configuration of the $.ApplyConfig.Type|public$ type for use\n", typeParams)
	sw.Do("// with apply.\n", typeParams)
	sw.Do("type $.ApplyConfig.ApplyConfiguration|public$ struct {\n", typeParams)

	for _, structMember := range typeParams.Struct.Members {
		if structMemberTags, ok := lookupJSONTags(structMember); ok {
			if structMemberTags.inline {
				params := memberParams{
					TypeParams: typeParams,
					Member:     structMember,
					MemberType: g.refGraph.applyConfigForType(structMember.Type),
					JSONTags:   structMemberTags,
				}
				sw.Do("$.MemberType|raw$ `json:\"$.JSONTags$\"`\n", params)
			} else {
				structMemberTags.omitempty = true
				params := memberParams{
					TypeParams: typeParams,
					Member:     structMember,
					MemberType: g.refGraph.applyConfigForType(structMember.Type),
					JSONTags:   structMemberTags,
				}
				sw.Do("$.Member.Name$ *$.MemberType|raw$ `json:\"$.JSONTags$\"`\n", params)
			}
		}
	}
	sw.Do("}\n", typeParams)
}

var listAlias = `
// $.ApplyConfig.Type|public$List represents a listAlias of $.ApplyConfig.ApplyConfiguration|public$.
type $.ApplyConfig.Type|public$List []*$.ApplyConfig.ApplyConfiguration|public$
`

var mapAlias = `
// $.ApplyConfig.Type|public$List represents a map of $.ApplyConfig.ApplyConfiguration|public$.
type $.ApplyConfig.Type|public$Map map[string]$.ApplyConfig.ApplyConfiguration|public$
`
