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

type TypeParams struct {
	Struct      *types.Type
	ApplyConfig applyConfig
	Refs        map[string]*types.Type
}

type memberParams struct {
	TypeParams
	Member     types.Member
	MemberType *types.Type
	JsonTags   jsonTags
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
	sw.Do(constructor, typeParams)

	for _, member := range t.Members {
		memberType := g.refGraph.applyConfigForType(member.Type)
		if jsonTags, ok := lookupJsonTags(member); ok {
			if g.refGraph.isApplyConfig(member.Type) {
				memberType = &types.Type{Kind: types.Pointer, Elem: memberType}
			}
			memberParams := memberParams{
				TypeParams: typeParams,
				Member:     member,
				MemberType: memberType,
				JsonTags:   jsonTags,
			}
			g.generateMemberSet(sw, memberParams)
			g.generateMemberRemove(sw, memberParams)
			g.generateMemberGet(sw, memberParams)
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
		if structMemberTags, ok := lookupJsonTags(structMember); ok {
			if structMemberTags.inline {
				params := memberParams{
					TypeParams: typeParams,
					Member:     structMember,
					MemberType: g.refGraph.applyConfigForType(structMember.Type),
					JsonTags:   structMemberTags,
				}
				sw.Do("$.MemberType|raw$ `json:\"$.JsonTags$\"`\n", params)
			} else {
				structMemberTags.omitempty = true
				params := memberParams{
					TypeParams: typeParams,
					Member:     structMember,
					MemberType: g.refGraph.applyConfigForType(structMember.Type),
					JsonTags:   structMemberTags,
				}
				sw.Do("$.Member.Name$ *$.MemberType|raw$ `json:\"$.JsonTags$\"`\n", params)
			}
		}
	}
	sw.Do("}\n", typeParams)
}

func (g *applyConfigurationGenerator) generateMemberSet(sw *generator.SnippetWriter, memberParams memberParams) {
	sw.Do("// Set$.Member.Name$ sets the $.Member.Name$ field in the declarative configuration to the given value.\n", memberParams)
	sw.Do("func (b *$.ApplyConfig.ApplyConfiguration|public$) Set$.Member.Name$(value $.MemberType|raw$) *$.ApplyConfig.ApplyConfiguration|public$ {\n", memberParams)
	if memberParams.JsonTags.inline {
		sw.Do("if value != nil {\n", memberParams)
		sw.Do("  b.$.Member.Name$ApplyConfiguration = *value\n", memberParams)
		sw.Do("}\n", memberParams)
	} else if g.refGraph.isApplyConfig(memberParams.Member.Type) {
		sw.Do("b.$.Member.Name$ = value\n", memberParams)
	} else {
		sw.Do("b.$.Member.Name$ = &value\n", memberParams)
	}
	sw.Do("  return b\n", memberParams)
	sw.Do("}\n", memberParams)
}

func (g *applyConfigurationGenerator) generateMemberRemove(sw *generator.SnippetWriter, memberParams memberParams) {
	if memberParams.JsonTags.inline {
		// Inline types cannot be removed
		return
	}
	sw.Do("// Remove$.Member.Name$ removes the $.Member.Name$ field from the declarative configuration.\n", memberParams)
	sw.Do("func (b *$.ApplyConfig.ApplyConfiguration|public$) Remove$.Member.Name$() *$.ApplyConfig.ApplyConfiguration|public$ {\n", memberParams)
	sw.Do("b.$.Member.Name$ = nil\n", memberParams)
	sw.Do("  return b\n", memberParams)
	sw.Do("}\n", memberParams)
}

func (g *applyConfigurationGenerator) generateMemberGet(sw *generator.SnippetWriter, memberParams memberParams) {
	sw.Do("// Get$.Member.Name$ gets the $.Member.Name$ field from the declarative configuration.\n", memberParams)
	sw.Do("func (b *$.ApplyConfig.ApplyConfiguration|public$) Get$.Member.Name$() (value $.MemberType|raw$, ok bool) {\n", memberParams)
	if memberParams.JsonTags.inline {
		sw.Do("return &b.$.Member.Name$ApplyConfiguration, true\n", memberParams)
	} else if g.refGraph.isApplyConfig(memberParams.Member.Type) {
		sw.Do("return b.$.Member.Name$, b.$.Member.Name$ != nil\n", memberParams)
	} else {
		sw.Do("if v := b.$.Member.Name$; v != nil {\n", memberParams)
		sw.Do("  return *v, true\n", memberParams)
		sw.Do("}\n", memberParams)
		sw.Do("return value, false\n", memberParams)
	}
	sw.Do("}\n", memberParams)
}

var constructor = `
// $.ApplyConfig.ApplyConfiguration|public$ constructs an declarative configuration of the $.ApplyConfig.Type|public$ type for use with
// apply.
func $.ApplyConfig.Type|public$() *$.ApplyConfig.ApplyConfiguration|public$ {
  return &$.ApplyConfig.ApplyConfiguration|public${}
}
`

var listAlias = `
// $.ApplyConfig.Type|public$List represents a listAlias of $.ApplyConfig.ApplyConfiguration|public$.
type $.ApplyConfig.Type|public$List []*$.ApplyConfig.ApplyConfiguration|public$
`

var mapAlias = `
// $.ApplyConfig.Type|public$List represents a map of $.ApplyConfig.ApplyConfiguration|public$.
type $.ApplyConfig.Type|public$Map map[string]$.ApplyConfig.ApplyConfiguration|public$
`
