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
	"sigs.k8s.io/structured-merge-diff/v4/merge"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

// portListParser sets the default value of key "protocol" to "TCP"
var portListParser = func() *typed.Parser {
	parser, err := typed.NewParser(`types:
- name: v1
  map:
    fields:
      - name: containerPorts
        type:
          list:
            elementType:
              map:
                fields:
                - name: port
                  type:
                    scalar: numeric
                - name: protocol
                  default: "TCP"
                  type:
                    scalar: string
                - name: name
                  type:
                    scalar: string
            elementRelationship: associative
            keys:
            - port
            - protocol
`)
	if err != nil {
		panic(err)
	}
	return parser
}()

func TestDefaultKeysFlat(t *testing.T) {
	tests := map[string]TestCase{
		"apply_missing_defaulted_key_A": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						containerPorts:
						- port: 80
					`,
				},
			},
			APIVersion: "v1",
			Object: `
				containerPorts:
				- port: 80
			`,
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("containerPorts", _KBF("port", 80, "protocol", "TCP")),
						_P("containerPorts", _KBF("port", 80, "protocol", "TCP"), "port"),
					),
					"v1",
					false,
				),
			},
		},
		"apply_missing_defaulted_key_B": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						containerPorts:
						- port: 80
						- port: 80
						  protocol: UDP
					`,
				},
			},
			APIVersion: "v1",
			Object: `
				containerPorts:
				- port: 80
				- port: 80
				  protocol: UDP
			`,
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("containerPorts", _KBF("port", 80, "protocol", "TCP")),
						_P("containerPorts", _KBF("port", 80, "protocol", "TCP"), "port"),
						_P("containerPorts", _KBF("port", 80, "protocol", "UDP")),
						_P("containerPorts", _KBF("port", 80, "protocol", "UDP"), "port"),
						_P("containerPorts", _KBF("port", 80, "protocol", "UDP"), "protocol"),
					),
					"v1",
					false,
				),
			},
		},
		"apply_missing_defaulted_key_with_conflict": {
			Ops: []Operation{
				Apply{
					Manager:    "apply-one",
					APIVersion: "v1",
					Object: `
						containerPorts:
						- port: 80
						  protocol: TCP
						  name: foo
					`,
				},
				Apply{
					Manager:    "apply-two",
					APIVersion: "v1",
					Object: `
						containerPorts:
						- port: 80
						  name: bar
					`,
					Conflicts: merge.Conflicts{
						merge.Conflict{Manager: "apply-one", Path: _P("containerPorts", _KBF("port", 80, "protocol", "TCP"), "name")},
					},
				},
			},
			APIVersion: "v1",
			Object: `
				containerPorts:
				- port: 80
				  protocol: TCP
				  name: foo
			`,
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("containerPorts", _KBF("port", 80, "protocol", "TCP")),
						_P("containerPorts", _KBF("port", 80, "protocol", "TCP"), "port"),
						_P("containerPorts", _KBF("port", 80, "protocol", "TCP"), "protocol"),
						_P("containerPorts", _KBF("port", 80, "protocol", "TCP"), "name"),
					),
					"v1",
					false,
				),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.Test(portListParser); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestDefaultKeysFlatErrors(t *testing.T) {
	tests := map[string]TestCase{
		"apply_missing_undefaulted_defaulted_key": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						containerPorts:
						- protocol: TCP
					`,
				},
			},
		},
		"apply_missing_defaulted_key_ambiguous_A": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						containerPorts:
						- port: 80
						- port: 80
					`,
				},
			},
		},
		"apply_missing_defaulted_key_ambiguous_B": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						containerPorts:
						- port: 80
						- port: 80
						  protocol: TCP
					`,
				},
			},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if test.Test(portListParser) == nil {
				t.Fatal("Should fail")
			}
		})
	}
}

// bookParser sets the default value of key:
// * "chapter" to 1
// * "section" to "A"
// * "page" to 2,
// * "line" to 3,
var bookParser = func() *typed.Parser {
	parser, err := typed.NewParser(`types:
- name: v1
  map:
    fields:
      - name: book
        type:
          list:
            elementType:
              map:
                fields:
                - name: chapter
                  default: 1
                  type:
                    scalar: numeric
                - name: section
                  default: "A"
                  type:
                    scalar: string
                - name: sentences
                  type:
                    list:
                      elementType:
                        map:
                          fields:
                          - name: page
                            default: 2.0
                            type:
                              scalar: numeric
                          - name: line
                            default: 3
                            type:
                              scalar: numeric
                          - name: text
                            type:
                              scalar: string
                      elementRelationship: associative
                      keys:
                      - page
                      - line
            elementRelationship: associative
            keys:
            - chapter
            - section
`)
	if err != nil {
		panic(err)
	}
	return parser
}()

func TestDefaultKeysNested(t *testing.T) {
	tests := map[string]TestCase{
		"apply_missing_every_key_nested": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						book:
						- sentences:
						  - text: blah
					`,
				},
			},
			APIVersion: "v1",
			Object: `
				book:
				- sentences:
				  - text: blah
			`,
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P(
							"book", _KBF("chapter", 1, "section", "A"),
						),
						_P(
							"book", _KBF("chapter", 1, "section", "A"),
							"sentences", _KBF("page", 2, "line", 3),
						),
						_P(
							"book", _KBF("chapter", 1, "section", "A"),
							"sentences", _KBF("page", 2, "line", 3),
							"text",
						),
					),
					"v1",
					false,
				),
			},
		},
		"apply_integer_key_with_float_default": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						book:
						- sentences:
						  - text: blah
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						book:
						- sentences:
						  - text: blah
						    page: 2
					`,
				},
			},
			APIVersion: "v1",
			Object: `
				book:
				- sentences:
				  - text: blah
				    page: 2
			`,
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P(
							"book", _KBF("chapter", 1, "section", "A"),
						),
						_P(
							"book", _KBF("chapter", 1, "section", "A"),
							"sentences", _KBF("page", 2, "line", 3),
						),
						_P(
							"book", _KBF("chapter", 1, "section", "A"),
							"sentences", _KBF("page", 2, "line", 3),
							"text",
						),
						_P(
							"book", _KBF("chapter", 1, "section", "A"),
							"sentences", _KBF("page", 2, "line", 3),
							"page",
						),
					),
					"v1",
					false,
				),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.Test(bookParser); err != nil {
				t.Fatal(err)
			}
		})
	}
}
