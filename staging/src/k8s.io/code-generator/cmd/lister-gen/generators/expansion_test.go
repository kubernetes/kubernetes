/*
Copyright 2023 The Kubernetes Authors.

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
	"reflect"
	"testing"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

var mockType = &types.Type{
	Name: types.Name{
		Package: "kubernetes/staging/src/k8s.io/code-generator/cmd/lister-gen/generators",
		Name: "generators",
	},
	CommentLines: []string{
		"RandomType defines a random structure in Kubernetes",
		"It should be used just when you need something different than 42",
	},
	SecondClosestCommentLines: []string{},
}

func TestFilter(t *testing.T) {
	g := &expansionGenerator{
		types: []*types.Type{
			mockType,
		},
	}
	tests := []struct {
		name	string
		context *generator.Context
		input 	*types.Type
		want 	bool
	}{
		{
			name: "ContextTest",
			context: &generator.Context{
				Namers: make(namer.NameSystems),
				Universe: types.Universe{},
				Inputs: []string{"package1", "package2"},
				Verify: true,
			},
			input: &types.Type{},
			want: true,
		},
		{
			name: "No Inputs",
			context: &generator.Context{
				Namers: make(namer.NameSystems),
				Universe: types.Universe{},
				Verify: true,
			},
			input: &types.Type{},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name , func(t *testing.T) {
			got := g.Filter(tt.context, mockType)
			if !reflect.DeepEqual(tt.want, got) {
				t.Errorf("Filter(%v %v) = %v, want %v",tt.context, tt.input, got, tt.want)
			}
			if tt.context.Namers == nil {
				t.Errorf("Namers is nil, expected a map")
			}
			if tt.context.Universe == nil {
				t.Errorf("Universe is nil, expected a Universe instance")
			}
			if tt.context.Inputs != nil {
				expectedInput := []string{"package1", "package2"}
				if !reflect.DeepEqual(expectedInput, tt.context.Inputs) {
					t.Errorf("Received mismatched inputs")
				}
			}
		})
	}
}
