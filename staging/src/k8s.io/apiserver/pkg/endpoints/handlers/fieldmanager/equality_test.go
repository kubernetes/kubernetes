/*
Copyright 20214The Kubernetes Authors.

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

package fieldmanager

import "testing"

func TestEqualIgnoringFieldValueAtPath(t *testing.T) {
	cases := []struct {
		name string
		a, b map[string]any
		want bool
	}{
		{
			name: "identical objects",
			a: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "dev"},
					"managedFields": []any{},
				},
				"spec": map[string]any{
					"field": "value",
				},
			},
			b: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "dev"},
					"managedFields": []any{},
				},
				"spec": map[string]any{
					"field": "value",
				},
			},
			want: true,
		},
		{
			name: "different metadata label value",
			a: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "dev"},
					"managedFields": []any{},
				},
				"spec": map[string]any{
					"field": "value",
				},
			},
			b: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "prod"},
					"managedFields": []any{},
				},
				"spec": map[string]any{
					"field": "value",
				},
			},
			want: false,
		},
		{
			name: "different spec value",
			a: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "dev"},
					"managedFields": []any{},
				},
				"spec": map[string]any{
					"field": "value1",
				},
			},
			b: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "dev"},
					"managedFields": []any{},
				},
				"spec": map[string]any{
					"field": "value2",
				},
			},
			want: false,
		},
		{
			name: "extra spec fields in object a",
			a: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "dev"},
					"managedFields": []any{},
				},
				"spec": map[string]any{
					"field":      "value1",
					"otherField": "other",
				},
			},
			b: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "dev"},
					"managedFields": []any{},
				},
				"spec": map[string]any{
					"field": "value1",
				},
			},
			want: false,
		},
		{
			name: "different spec field in object b",
			a: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "dev"},
					"managedFields": []any{},
				},
				"spec": map[string]any{
					"field": "value1",
				},
			},
			b: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "dev"},
					"managedFields": []any{},
				},
				"spec": map[string]any{
					"field":      "value1",
					"otherField": "other",
				},
			},
			want: false,
		},
		{
			name: "different managed fields should be ignored",
			a: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "dev"},
					"managedFields": []any{map[string]any{"manager": "client1"}},
				},
				"spec": map[string]any{
					"field": "value",
				},
			},
			b: map[string]any{
				"metadata": map[string]any{
					"labels":        map[string]any{"env": "dev"},
					"managedFields": []any{map[string]any{"manager": "client2"}},
				},
				"spec": map[string]any{
					"field": "value",
				},
			},
			want: true,
		},
	}

	path := []string{"metadata", "managedFields"}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			actual := equalIgnoringValueAtPath(c.a, c.b, path)
			if actual != c.want {
				t.Error("Expected equality check to return ", c.want, ", but got ", actual)
			}
		})
	}
}
