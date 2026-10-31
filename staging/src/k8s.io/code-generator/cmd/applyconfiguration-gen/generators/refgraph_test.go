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

package generators

import (
	"testing"

	"k8s.io/gengo/v2/types"
)

func TestApplyConfigForTypeUsesPublicName(t *testing.T) {
	for _, name := range []string{"common", "Common"} {
		t.Run(name, func(t *testing.T) {
			field := &types.Type{
				Name: types.Name{Package: "example.com/api/v1", Name: name},
				Kind: types.Struct,
			}
			refs := refGraph{field.Name: "example.com/applyconfiguration/api/v1"}

			got := refs.applyConfigForType(field)
			want := types.Name{Package: "example.com/applyconfiguration/api/v1", Name: "CommonApplyConfiguration"}
			if got.Name != want {
				t.Errorf("got %v, want %v", got.Name, want)
			}
		})
	}
}
