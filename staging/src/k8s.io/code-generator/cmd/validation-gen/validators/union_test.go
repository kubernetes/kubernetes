/*
Copyright The Kubernetes Authors.

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

package validators

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/gengo/v2/types"
)

func TestUnionEmissionPaths(t *testing.T) {
	for _, discriminated := range []bool{false, true} {
		name := "union"
		if discriminated {
			name = "discriminated union"
		}
		t.Run(name, func(t *testing.T) {
			u := newUnion()
			u.members = []unionMember{{fieldName: "first", discriminatorValue: "First"}, {fieldName: "second", discriminatorValue: "Second"}}
			u.fieldMembers = []*types.Member{
				{Name: "First", Type: types.PointerTo(types.String)},
				{Name: "Second", Type: types.PointerTo(types.String)},
			}
			want := []Emission{{Type: field.ErrorTypeInvalid, Origin: "union"}}
			if discriminated {
				discriminator := "type"
				u.discriminator = &discriminator
				u.discriminatorMember = &types.Member{Name: "Type", Type: types.String}
				want = []Emission{
					{Type: field.ErrorTypeInvalid, Origin: "union", PathFragment: ".first"},
					{Type: field.ErrorTypeInvalid, Origin: "union", PathFragment: ".second"},
				}
			}
			parent := &types.Type{Kind: types.Struct, Name: types.Name{Name: "Example"}}
			got, err := processUnionValidations(field.NewPath("Example"), parent, unions{"": u}, "unionMembership", unionMemberTagName, unionValidator, discriminatedUnionValidator, Emission{Type: field.ErrorTypeInvalid, Origin: "union"})
			if err != nil {
				t.Fatal(err)
			}
			if len(got.Functions) != 1 {
				t.Fatalf("expected one union validation, got %d", len(got.Functions))
			}
			if !reflect.DeepEqual(got.Functions[0].Emits, want) {
				t.Errorf("emission paths: got %#v, want %#v", got.Functions[0].Emits, want)
			}
		})
	}
}
