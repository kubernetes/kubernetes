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
	"testing"

	"k8s.io/gengo/v2/types"
)

func TestRequirednessTagsOnNonPointerStructsAreAccepted(t *testing.T) {
	st := &types.Type{Kind: types.Struct, Name: types.Name{Name: "MyStruct"}}
	member := &types.Member{Name: "field", Type: st, CommentLines: nil}

	t.Run("required", func(t *testing.T) {
		rtv := requirednessTagValidator{requirednessRequired}
		v, err := rtv.doRequired(Context{Scope: ScopeField, Type: st, Member: member})
		if err != nil {
			t.Fatalf("doRequired() unexpected error: %v", err)
		}
		if len(v.Functions) != 0 {
			t.Fatalf("doRequired() expected no runtime validations for non-pointer struct, got %d", len(v.Functions))
		}
		if len(v.Comments) == 0 {
			t.Fatalf("doRequired() expected documentation comment for non-pointer struct")
		}
	})

	t.Run("optional", func(t *testing.T) {
		rtv := requirednessTagValidator{requirednessOptional}
		v, err := rtv.doOptional(Context{Scope: ScopeField, Type: st, Member: member})
		if err != nil {
			t.Fatalf("doOptional() unexpected error: %v", err)
		}
		if len(v.Functions) != 0 {
			t.Fatalf("doOptional() expected no runtime validations for non-pointer struct, got %d", len(v.Functions))
		}
		if len(v.Comments) == 0 {
			t.Fatalf("doOptional() expected documentation comment for non-pointer struct")
		}
	})
}
