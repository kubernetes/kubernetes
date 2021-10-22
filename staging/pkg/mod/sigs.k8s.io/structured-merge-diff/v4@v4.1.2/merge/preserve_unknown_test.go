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

package merge_test

import (
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	. "sigs.k8s.io/structured-merge-diff/v4/internal/fixture"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

var preserveUnknownParser = func() Parser {
	parser, err := typed.NewParser(`types:
- name: type
  map:
    fields:
      - name: num
        type:
          scalar: numeric
    elementType:
      scalar: string
`)
	if err != nil {
		panic(err)
	}
	return SameVersionParser{T: parser.Type("type")}
}()

func TestPreserveUnknownFields(t *testing.T) {
	tests := map[string]TestCase{
		"preserve_unknown_fields": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						num: 5
						unknown: value
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						num: 6
						unknown: new
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				num: 6
				unknown: new
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("num"),
						_P("unknown"),
					),
					"v1",
					false,
				),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.Test(preserveUnknownParser); err != nil {
				t.Fatal(err)
			}
		})
	}
}
