/*
Copyright 2026 The Kubernetes Authors.

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

package protobuf

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

const (
	package1       = "github.com/user/pkg1"
	package2       = "github.com/user/pkg2"
	unknownPackage = "github.com/user/unknown"
)

var TypeMetadata = types.Name{
	Package: "",
	Name:    "",
	Path:    "",
}

var protoPackages = []*protobufPackage{
	newProtobufPackage(package1, "vendor/github.com/user/pkg1", "pkg1", true, map[types.Name]struct{}{TypeMetadata: {}}),
	newProtobufPackage(package2, "vendor/github.com/user/pkg2", "pkg2", true, map[types.Name]struct{}{TypeMetadata: {}}),
}

func prepareTestData(packages ...string) []*types.Type {
	var typeList []*types.Type
	for _, pkg := range packages {
		innerStruct := &types.Type{
			Name: types.Name{Package: pkg, Name: "StructMember"},
			Kind: types.Struct,
			Members: []types.Member{
				{
					Name:     "Field1",
					Embedded: false,
					Type: &types.Type{
						Name: types.Name{Package: "", Name: "string"},
						Kind: types.Builtin,
					},
					Tags: `json:",inline"`,
				},
				{
					Name:     "Field2",
					Embedded: false,
					Type: &types.Type{
						Name: types.Name{Package: "", Name: "string"},
						Kind: types.Builtin,
					},
					Tags: `json:",inline"`,
				},
			},
		}

		memberListWithEmbeddedMembers := []types.Member{
			{
				Name:     "ObjectMeta",
				Embedded: true,
				Type:     innerStruct,
				Tags:     `json:",inline"`,
			},
			{
				Name:     "StructMember",
				Embedded: true,
				Type:     innerStruct,
				Tags:     `json:",inline"`,
			},
			{
				Name:     "StringMember",
				Embedded: false,
				Type: &types.Type{
					Name: types.Name{Package: "", Name: "string"},
					Kind: types.Builtin,
				},
				Tags: `json:"stringMember,omitempty"`,
			},
		}

		t := &types.Type{
			Name: types.Name{
				Package: pkg,
				Name:    "Spec",
			},
			Kind:    types.Struct,
			Members: memberListWithEmbeddedMembers,
		}
		typeList = append(typeList, t)
	}
	return typeList
}

func TestFlattenEmbeddedMembersInPackages(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                          string
		ctx                           *generator.Context
		AllowedValuesForEmbeddedField []bool
	}{
		{
			name: "Struct with Embedded fields must be flattened",
			ctx: &generator.Context{
				Order: prepareTestData(package1, package2),
			},
			AllowedValuesForEmbeddedField: []bool{false},
		},
		{
			name: "Struct with Embedded fields in an Unknown Packages must be untouched",
			ctx: &generator.Context{
				Order: prepareTestData(unknownPackage),
			},
			AllowedValuesForEmbeddedField: []bool{true, false},
		},
	}
	var validatorFunc func(flattener *flattener, members []types.Member, AllowedValuesForEmbeddedField []bool) bool
	validatorFunc = func(flattener *flattener, members []types.Member, AllowedValuesForEmbeddedField []bool) bool {
		for _, member := range members {
			// skip validation for ignored members
			if _, ok := flattener.ignoredMembers[member.Name]; ok {
				continue
			}
			if !slices.Contains(AllowedValuesForEmbeddedField, member.Embedded) {
				return false
			}
			if !validatorFunc(flattener, member.Type.Members, AllowedValuesForEmbeddedField) {
				return false
			}
		}
		return true
	}

	flattener := NewFlattener(protoPackages)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			flattener.flattenEmbeddedMembersInPackages(test.ctx)
			for _, member := range test.ctx.Order {
				assert.True(t, validatorFunc(flattener, member.Members, test.AllowedValuesForEmbeddedField), "Test failed")
			}
		})
	}
}
