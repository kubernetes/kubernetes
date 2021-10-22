/*
Copyright 2019 The Kubernetes Authors.

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

package typed_test

import (
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

var unionParser = func() typed.ParseableType {
	parser, err := typed.NewParser(`types:
- name: union
  map:
    fields:
    - name: discriminator
      type:
        scalar: string
    - name: one
      type:
        scalar: numeric
    - name: two
      type:
        scalar: numeric
    - name: three
      type:
        scalar: numeric
    - name: letter
      type:
        scalar: string
    - name: a
      type:
        scalar: numeric
    - name: b
      type:
        scalar: numeric
    unions:
    - discriminator: discriminator
      deduceInvalidDiscriminator: true
      fields:
      - fieldName: one
        discriminatorValue: One
      - fieldName: two
        discriminatorValue: TWO
      - fieldName: three
        discriminatorValue: three
    - discriminator: letter
      fields:
      - fieldName: a
        discriminatorValue: A
      - fieldName: b
        discriminatorValue: b`)
	if err != nil {
		panic(err)
	}
	return parser.Type("union")
}()

func TestNormalizeUnions(t *testing.T) {
	tests := []struct {
		name string
		old  typed.YAMLObject
		new  typed.YAMLObject
		out  typed.YAMLObject
	}{
		{
			name: "nothing changed, add discriminator",
			old:  `{"one": 1}`,
			new:  `{"one": 1}`,
			out:  `{"one": 1, "discriminator": "One"}`,
		},
		{
			name: "nothing changed, non-deduced",
			old:  `{"a": 1}`,
			new:  `{"a": 1}`,
			out:  `{"a": 1}`,
		},
		{
			name: "proper union update, setting discriminator",
			old:  `{"one": 1}`,
			new:  `{"two": 1}`,
			out:  `{"two": 1, "discriminator": "TWO"}`,
		},
		{
			name: "proper union update, non-deduced",
			old:  `{"a": 1}`,
			new:  `{"b": 1}`,
			out:  `{"b": 1}`,
		},
		{
			name: "proper union update from not-set, setting discriminator",
			old:  `{}`,
			new:  `{"two": 1}`,
			out:  `{"two": 1, "discriminator": "TWO"}`,
		},
		{
			name: "proper union update from not-set, non-deduced",
			old:  `{}`,
			new:  `{"b": 1}`,
			out:  `{"b": 1}`,
		},
		{
			name: "remove union, with discriminator",
			old:  `{"one": 1}`,
			new:  `{}`,
			out:  `{}`,
		},
		{
			name: "remove union and discriminator",
			old:  `{"one": 1, "discriminator": "One"}`,
			new:  `{}`,
			out:  `{}`,
		},
		{
			name: "remove union, not discriminator",
			old:  `{"one": 1, "discriminator": "One"}`,
			new:  `{"discriminator": "One"}`,
			out:  `{"discriminator": "One"}`,
		},
		{
			name: "remove union, not discriminator, non-deduced",
			old:  `{"a": 1, "letter": "A"}`,
			new:  `{"letter": "A"}`,
			out:  `{"letter": "A"}`,
		},
		{
			name: "change discriminator, nothing else",
			old:  `{"discriminator": "One"}`,
			new:  `{"discriminator": "random"}`,
			out:  `{"discriminator": "random"}`,
		},
		{
			name: "change discriminator, nothing else, non-deduced",
			old:  `{"letter": "A"}`,
			new:  `{"letter": "b"}`,
			out:  `{"letter": "b"}`,
		},
		{
			name: "change discriminator, nothing else, it drops other field",
			old:  `{"discriminator": "One", "one": 1}`,
			new:  `{"discriminator": "random", "one": 1}`,
			out:  `{"discriminator": "random"}`,
		},
		{
			name: "change discriminator, nothing else, it drops other field, non-deduced",
			old:  `{"letter": "A", "a": 1}`,
			new:  `{"letter": "b", "a": 1}`,
			out:  `{"letter": "b"}`,
		},
		{
			name: "remove discriminator, nothing else",
			old:  `{"discriminator": "One", "one": 1}`,
			new:  `{"one": 1}`,
			out:  `{"one": 1, "discriminator": "One"}`,
		},
		{
			name: "remove discriminator, nothing else, non-deduced",
			old:  `{"letter": "A", "a": 1}`,
			new:  `{"a": 1}`,
			out:  `{"a": 1}`,
		},
		{
			name: "remove discriminator, add new field",
			old:  `{"discriminator": "One", "one": 1}`,
			new:  `{"two": 1}`,
			out:  `{"two": 1, "discriminator": "TWO"}`,
		},
		{
			name: "remove discriminator, add new field, non-deduced",
			old:  `{"letter": "A", "a": 1}`,
			new:  `{"b": 1}`,
			out:  `{"b": 1}`,
		},
		{
			name: "both fields removed",
			old:  `{"one": 1, "two": 1}`,
			new:  `{}`,
			out:  `{}`,
		},
		{
			name: "one field removed",
			old:  `{"one": 1, "two": 1}`,
			new:  `{"one": 1}`,
			out:  `{"one": 1, "discriminator": "One"}`,
		},
		{
			name: "one field removed, non-deduced",
			old:  `{"a": 1, "b": 1}`,
			new:  `{"a": 1}`,
			out:  `{"a": 1}`,
		},
		// These use-cases shouldn't happen:
		{
			name: "one field removed, discriminator unchanged",
			old:  `{"one": 1, "two": 1, "discriminator": "TWO"}`,
			new:  `{"one": 1, "discriminator": "TWO"}`,
			out:  `{"one": 1, "discriminator": "One"}`,
		},
		{
			name: "one field removed, discriminator unchanged, non-deduced",
			old:  `{"a": 1, "b": 1, "letter": "b"}`,
			new:  `{"a": 1, "letter": "b"}`,
			out:  `{"a": 1, "letter": "b"}`,
		},
		{
			name: "one field removed, discriminator added",
			old:  `{"two": 2, "one": 1}`,
			new:  `{"one": 1, "discriminator": "TWO"}`,
			out:  `{"discriminator": "TWO"}`,
		},
		{
			name: "one field removed, discriminator added, non-deduced",
			old:  `{"b": 2, "a": 1}`,
			new:  `{"a": 1, "letter": "b"}`,
			out:  `{"letter": "b"}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			old, err := unionParser.FromYAML(test.old)
			if err != nil {
				t.Fatalf("Failed to parse old object: %v", err)
			}
			new, err := unionParser.FromYAML(test.new)
			if err != nil {
				t.Fatalf("failed to parse new object: %v", err)
			}
			out, err := unionParser.FromYAML(test.out)
			if err != nil {
				t.Fatalf("failed to parse out object: %v", err)
			}
			got, err := old.NormalizeUnions(new)
			if err != nil {
				t.Fatalf("failed to normalize unions: %v", err)
			}
			comparison, err := out.Compare(got)
			if err != nil {
				t.Fatalf("failed to compare result and expected: %v", err)
			}
			if !comparison.IsSame() {
				t.Errorf("Result is different from expected:\n%v", comparison)
			}
		})
	}
}

func TestNormalizeUnionError(t *testing.T) {
	tests := []struct {
		name string
		old  typed.YAMLObject
		new  typed.YAMLObject
	}{
		{
			name: "dumb client update, no discriminator",
			old:  `{"one": 1}`,
			new:  `{"one": 2, "two": 1}`,
		},
		{
			name: "new object has three of same union set",
			old:  `{"one": 1}`,
			new:  `{"one": 2, "two": 1, "three": 3}`,
		},
		{
			name: "dumb client doesn't update discriminator",
			old:  `{"one": 1, "discriminator": "One"}`,
			new:  `{"one": 2, "two": 1, "discriminator": "One"}`,
		},
		{
			name: "client sends new field that and discriminator change",
			old:  `{}`,
			new:  `{"one": 1, "discriminator": "Two"}`,
		},
		{
			name: "client sends new fields that don't match discriminator change",
			old:  `{}`,
			new:  `{"one": 1, "two": 1, "discriminator": "One"}`,
		},
		{
			name: "old object has two of same union set",
			old:  `{"one": 1, "two": 2}`,
			new:  `{"one": 2, "two": 1}`,
		},
		{
			name: "old object has two of same union, but we add third",
			old:  `{"discriminator": "One", "one": 1, "two": 1}`,
			new:  `{"discriminator": "One", "one": 1, "two": 1, "three": 1}`,
		},
		{
			name: "one field removed, 2 left, discriminator unchanged",
			old:  `{"one": 1, "two": 1, "three": 1, "discriminator": "TWO"}`,
			new:  `{"one": 1, "two": 1, "discriminator": "TWO"}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			old, err := unionParser.FromYAML(test.old)
			if err != nil {
				t.Fatalf("Failed to parse old object: %v", err)
			}
			new, err := unionParser.FromYAML(test.new)
			if err != nil {
				t.Fatalf("failed to parse new object: %v", err)
			}
			_, err = old.NormalizeUnions(new)
			if err == nil {
				t.Fatal("Normalization should have failed, but hasn't.")
			}
		})
	}
}
