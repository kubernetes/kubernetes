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
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

type flattener struct {
	packages []*protobufPackage
	// Map of Members for which Embedding is allowed
	ignoredMembers map[string]struct{}
}

func NewFlattener(packages []*protobufPackage) *flattener {
	return &flattener{
		packages: packages,
		ignoredMembers: map[string]struct{}{
			"ObjectMeta": {},
			"ListMeta":   {},
		},
	}
}

func (f *flattener) flattenEmbeddedMembersInPackages(c *generator.Context) {
	for _, p := range f.packages {
		for _, t := range c.Order {
			if t.Name.Package != p.Path() {
				continue
			}
			if !isTypeApplicableToProtobuf(t) {
				// skip types that we don't care about, like functions etc
				continue
			}
			f.flattenEmbeddedMembersInPackage(p, t)
		}
	}
}

func (f *flattener) flattenEmbeddedMembersInPackage(p *protobufPackage, t *types.Type) {

	// return if it is not a struct
	if t.Kind != types.Struct {
		return
	}

	var newMembers []types.Member
	for _, member := range t.Members {
		_, ok := p.OmitFieldTypes[member.Type.Name]
		if ok {
			continue
		}

		// Retain the member:
		// 1. If it is a non-embedded member
		// 2. If it is present in the ignoredMembers list
		if _, ok := f.ignoredMembers[member.Type.Name.Name]; ok || !member.Embedded {
			newMembers = append(newMembers, member)
			continue
		}

		// Process the embedded fields
		embeddedType := member.Type

		// Handles the pointer to embedded struct (example *SomeStruct -> SomeStruct)
		if embeddedType.Kind == types.Pointer {
			embeddedType = embeddedType.Elem
		}

		// Move to the next member if embedded field is not a struct
		if embeddedType.Kind != types.Struct {
			continue
		}

		// Proceed with flattening the inner structs first
		f.flattenEmbeddedMembersInPackage(p, embeddedType)

		for _, innerMember := range embeddedType.Members {
			clonedInnerMember := types.Member{
				Name:         innerMember.Name,
				Type:         innerMember.Type,
				Tags:         innerMember.Tags,
				CommentLines: innerMember.CommentLines,
				Embedded:     false,
			}
			newMembers = append(newMembers, clonedInnerMember)
		}
	}
	t.Members = newMembers
}
