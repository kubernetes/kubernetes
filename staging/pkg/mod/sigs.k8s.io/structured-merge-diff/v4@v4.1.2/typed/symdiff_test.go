/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

type symdiffTestCase struct {
	name         string
	rootTypeName string
	schema       typed.YAMLObject
	quints       []symdiffQuint
}

type symdiffQuint struct {
	lhs typed.YAMLObject
	rhs typed.YAMLObject

	// Please note that everything is tested both ways--removed and added
	// are symmetric. So if a test case is covered for one of them, it
	// covers both.
	removed  *fieldpath.Set
	modified *fieldpath.Set
	added    *fieldpath.Set
}

var symdiffCases = []symdiffTestCase{{
	name:         "simple pair",
	rootTypeName: "stringPair",
	schema: `types:
- name: stringPair
  map:
    fields:
    - name: key
      type:
        scalar: string
    - name: value
      type:
        namedType: __untyped_atomic_
- name: __untyped_atomic_
  scalar: untyped
  list:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
  map:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
`,
	quints: []symdiffQuint{{
		lhs:      `{"key":"foo","value":1}`,
		rhs:      `{"key":"foo","value":1}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(),
	}, {
		lhs:      `{"key":"foo","value":{}}`,
		rhs:      `{"key":"foo","value":1}`,
		removed:  _NS(),
		modified: _NS(_P("value")),
		added:    _NS(),
	}, {
		lhs:      `{"key":"foo","value":1}`,
		rhs:      `{"key":"foo","value":{}}`,
		removed:  _NS(),
		modified: _NS(_P("value")),
		added:    _NS(),
	}, {
		lhs:      `{"key":"foo","value":1}`,
		rhs:      `{"key":"foo","value":{"doesn't matter":"what's here","or":{"how":"nested"}}}`,
		removed:  _NS(),
		modified: _NS(_P("value")),
		added:    _NS(),
	}, {
		lhs:      `{"key":"foo","value":null}`,
		rhs:      `{"key":"foo","value":{}}`,
		removed:  _NS(),
		modified: _NS(_P("value")),
		added:    _NS(),
	}, {
		lhs:      `{"key":"foo"}`,
		rhs:      `{"value":true}`,
		removed:  _NS(_P("key")),
		modified: _NS(),
		added:    _NS(_P("value")),
	}, {
		lhs:      `{"key":"foot"}`,
		rhs:      `{"key":"foo","value":true}`,
		removed:  _NS(),
		modified: _NS(_P("key")),
		added:    _NS(_P("value")),
	}},
}, {
	name:         "null/empty map",
	rootTypeName: "nestedMap",
	schema: `types:
- name: nestedMap
  map:
    fields:
    - name: inner
      type:
        map:
          elementType:
            namedType: __untyped_atomic_
- name: __untyped_atomic_
  scalar: untyped
  list:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
  map:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
`,
	quints: []symdiffQuint{{
		lhs:      `{}`,
		rhs:      `{"inner":{}}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("inner")),
	}, {
		lhs:      `{}`,
		rhs:      `{"inner":null}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("inner")),
	}, {
		lhs:      `{"inner":null}`,
		rhs:      `{"inner":{}}`,
		removed:  _NS(),
		modified: _NS(_P("inner")),
		added:    _NS(),
	}, {
		lhs:      `{"inner":{}}`,
		rhs:      `{"inner":null}`,
		removed:  _NS(),
		modified: _NS(_P("inner")),
		added:    _NS(),
	}, {
		lhs:      `{"inner":{}}`,
		rhs:      `{"inner":{}}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(),
	}},
}, {
	name:         "null/empty struct",
	rootTypeName: "nestedStruct",
	schema: `types:
- name: nestedStruct
  map:
    fields:
    - name: inner
      type:
        map:
          fields:
          - name: value
            type:
              namedType: __untyped_atomic_
`,
	quints: []symdiffQuint{{
		lhs:      `{}`,
		rhs:      `{"inner":{}}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("inner")),
	}, {
		lhs:      `{}`,
		rhs:      `{"inner":null}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("inner")),
	}, {
		lhs:      `{"inner":null}`,
		rhs:      `{"inner":{}}`,
		removed:  _NS(),
		modified: _NS(_P("inner")),
		added:    _NS(),
	}, {
		lhs:      `{"inner":{}}`,
		rhs:      `{"inner":null}`,
		removed:  _NS(),
		modified: _NS(_P("inner")),
		added:    _NS(),
	}, {
		lhs:      `{"inner":{}}`,
		rhs:      `{"inner":{}}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(),
	}},
}, {
	name:         "null/empty list",
	rootTypeName: "nestedList",
	schema: `types:
- name: nestedList
  map:
    fields:
    - name: inner
      type:
        list:
          elementType:
            namedType: __untyped_atomic_
          elementRelationship: atomic
- name: __untyped_atomic_
  scalar: untyped
  list:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
  map:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
`,
	quints: []symdiffQuint{{
		lhs:      `{}`,
		rhs:      `{"inner":[]}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("inner")),
	}, {
		lhs:      `{}`,
		rhs:      `{"inner":null}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("inner")),
	}, {
		lhs:      `{"inner":null}`,
		rhs:      `{"inner":[]}`,
		removed:  _NS(),
		modified: _NS(_P("inner")),
		added:    _NS(),
	}, {
		lhs:      `{"inner":[]}`,
		rhs:      `{"inner":null}`,
		removed:  _NS(),
		modified: _NS(_P("inner")),
		added:    _NS(),
	}, {
		lhs:      `{"inner":[]}`,
		rhs:      `{"inner":[]}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(),
	}},
}, {
	name:         "map merge",
	rootTypeName: "nestedMap",
	schema: `types:
- name: nestedMap
  map:
    elementType:
      namedType: nestedMap
`,
	quints: []symdiffQuint{{
		lhs:      `{"a":{},"b":{}}`,
		rhs:      `{"a":{},"b":{}}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(),
	}, {
		lhs:      `{"a":{}}`,
		rhs:      `{"b":{}}`,
		removed:  _NS(_P("a")),
		modified: _NS(),
		added:    _NS(_P("b")),
	}, {
		lhs:      `{"a":{"b":{"c":{}}}}`,
		rhs:      `{"a":{"b":{}}}`,
		removed:  _NS(_P("a", "b", "c")),
		modified: _NS(),
		added:    _NS(),
	}, {
		lhs:      `{"a":{}}`,
		rhs:      `{"a":{"b":{}}}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("a", "b")),
	}},
}, {
	name:         "untyped deduced",
	rootTypeName: "__untyped_deduced_",
	schema: `types:
- name: __untyped_atomic_
  scalar: untyped
  list:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
  map:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
- name: __untyped_deduced_
  scalar: untyped
  list:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
  map:
    elementType:
      namedType: __untyped_deduced_
    elementRelationship: separable
`,
	quints: []symdiffQuint{{
		lhs:      `{"a":{}}}`,
		rhs:      `{"a":{"b":{}}}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("a", "b")),
	}, {
		lhs:      `{"a":null}`,
		rhs:      `{"a":{"b":{}}}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("a", "b")),
	}, {
		lhs:      `{"a":{"b":{}}}`,
		rhs:      `{"a":{}}}`,
		removed:  _NS(_P("a", "b")),
		modified: _NS(),
		added:    _NS(),
	}, {
		lhs:      `{"a":{"b":{}}}`,
		rhs:      `{"a":null}`,
		removed:  _NS(_P("a", "b")),
		modified: _NS(),
		added:    _NS(),
	}, {
		lhs:      `{"a":[]}`,
		rhs:      `{"a":["b"]}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":null}`,
		rhs:      `{"a":["b"]}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":["b"]}`,
		rhs:      `{"a":[]}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":["b"]}`,
		rhs:      `{"a":null}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":null}`,
		rhs:      `{"a":"b"}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":"b"}`,
		rhs:      `{"a":null}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":{"b":{}}}`,
		rhs:      `{"a":["b"]}}`,
		removed:  _NS(_P("a", "b")),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":["b"]}}`,
		rhs:      `{"a":{"b":{}}}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(_P("a", "b")),
	}, {
		lhs:      `{"a":{"b":{}}}`,
		rhs:      `{"a":"b"}`,
		removed:  _NS(_P("a", "b")),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":"b"}`,
		rhs:      `{"a":{"b":{}}}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(_P("a", "b")),
	}, {
		lhs:      `{"a":["b"]}}`,
		rhs:      `{"a":"b"}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":"b"}`,
		rhs:      `{"a":["b"]}}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(),
	}},
}, {
	name:         "untyped separable",
	rootTypeName: "__untyped_separable_",
	schema: `types:
- name: __untyped_separable_
  scalar: untyped
  list:
    elementType:
      namedType: __untyped_separable_
    elementRelationship: associative
  map:
    elementType:
      namedType: __untyped_separable_
    elementRelationship: separable
`,
	quints: []symdiffQuint{{
		lhs:      `{"a":{}}}`,
		rhs:      `{"a":{"b":{}}}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("a", "b")),
	}, {
		lhs:      `{"a":null}`,
		rhs:      `{"a":{"b":{}}}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("a", "b")),
	}, {
		lhs:      `{"a":{"b":{}}}`,
		rhs:      `{"a":{}}}`,
		removed:  _NS(_P("a", "b")),
		modified: _NS(),
		added:    _NS(),
	}, {
		lhs:      `{"a":{"b":{}}}`,
		rhs:      `{"a":null}`,
		removed:  _NS(_P("a", "b")),
		modified: _NS(),
		added:    _NS(),
	}, {
		lhs:      `{"a":[]}`,
		rhs:      `{"a":["b"]}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("a", _V("b"))),
	}, {
		lhs:     `{"a":null}`,
		rhs:     `{"a":["b"]}`,
		removed: _NS(),
		// TODO: result should be the same as the previous case
		// nothing shoule be modified here.
		modified: _NS(_P("a")),
		added:    _NS(_P("a", _V("b"))),
	}, {
		lhs:      `{"a":["b"]}`,
		rhs:      `{"a":[]}`,
		removed:  _NS(_P("a", _V("b"))),
		modified: _NS(),
		added:    _NS(),
	}, {
		lhs:     `{"a":["b"]}`,
		rhs:     `{"a":null}`,
		removed: _NS(_P("a", _V("b"))),
		// TODO: result should be the same as the previous case
		// nothing shoule be modified here.
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":null}`,
		rhs:      `{"a":"b"}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":"b"}`,
		rhs:      `{"a":null}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":{"b":{}}}`,
		rhs:      `{"a":["b"]}}`,
		removed:  _NS(_P("a", "b")),
		modified: _NS(),
		added:    _NS(_P("a", _V("b"))),
	}, {
		lhs:      `{"a":["b"]}}`,
		rhs:      `{"a":{"b":{}}}`,
		removed:  _NS(_P("a", _V("b"))),
		modified: _NS(),
		added:    _NS(_P("a", "b")),
	}, {
		lhs:      `{"a":{"b":{}}}`,
		rhs:      `{"a":"b"}`,
		removed:  _NS(_P("a", "b")),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":"b"}`,
		rhs:      `{"a":{"b":{}}}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(_P("a", "b")),
	}, {
		lhs:      `{"a":["b"]}}`,
		rhs:      `{"a":"b"}`,
		removed:  _NS(_P("a", _V("b"))),
		modified: _NS(_P("a")),
		added:    _NS(),
	}, {
		lhs:      `{"a":"b"}`,
		rhs:      `{"a":["b"]}}`,
		removed:  _NS(),
		modified: _NS(_P("a")),
		added:    _NS(_P("a", _V("b"))),
	}},
}, {
	name:         "struct grab bag",
	rootTypeName: "myStruct",
	schema: `types:
- name: myStruct
  map:
    fields:
    - name: numeric
      type:
        scalar: numeric
    - name: string
      type:
        scalar: string
    - name: bool
      type:
        scalar: boolean
    - name: setStr
      type:
        list:
          elementType:
            scalar: string
          elementRelationship: associative
    - name: setBool
      type:
        list:
          elementType:
            scalar: boolean
          elementRelationship: associative
    - name: setNumeric
      type:
        list:
          elementType:
            scalar: numeric
          elementRelationship: associative
`,
	quints: []symdiffQuint{{
		lhs:      `{"numeric":1}`,
		rhs:      `{"numeric":3.14159}`,
		removed:  _NS(),
		modified: _NS(_P("numeric")),
		added:    _NS(),
	}, {
		lhs:      `{"numeric":3.14159}`,
		rhs:      `{"numeric":1}`,
		removed:  _NS(),
		modified: _NS(_P("numeric")),
		added:    _NS(),
	}, {
		lhs:      `{"string":"aoeu"}`,
		rhs:      `{"bool":true}`,
		removed:  _NS(_P("string")),
		modified: _NS(),
		added:    _NS(_P("bool")),
	}, {
		lhs:      `{"setStr":["a","b"]}`,
		rhs:      `{"setStr":["a","b","c"]}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("setStr", _V("c"))),
	}, {
		lhs: `{"setStr":["a","b","c"]}`,
		rhs: `{"setStr":[]}`,
		removed: _NS(
			_P("setStr", _V("a")),
			_P("setStr", _V("b")),
			_P("setStr", _V("c")),
		),
		modified: _NS(),
		added:    _NS(),
	}, {
		lhs:      `{"setBool":[true]}`,
		rhs:      `{"setBool":[false]}`,
		removed:  _NS(_P("setBool", _V(true))),
		modified: _NS(),
		added:    _NS(_P("setBool", _V(false))),
	}, {
		lhs:      `{"setNumeric":[1,2,3.14159]}`,
		rhs:      `{"setNumeric":[1,2,3]}`,
		removed:  _NS(_P("setNumeric", _V(3.14159))),
		modified: _NS(),
		added:    _NS(_P("setNumeric", _V(3))),
	}},
}, {
	name:         "associative list",
	rootTypeName: "myRoot",
	schema: `types:
- name: myRoot
  map:
    fields:
    - name: list
      type:
        namedType: myList
    - name: atomicList
      type:
        namedType: mySequence
- name: myList
  list:
    elementType:
      namedType: myElement
    elementRelationship: associative
    keys:
    - key
    - id
- name: mySequence
  list:
    elementType:
      scalar: string
    elementRelationship: atomic
- name: myElement
  map:
    fields:
    - name: key
      type:
        scalar: string
    - name: id
      type:
        scalar: numeric
    - name: value
      type:
        namedType: myValue
    - name: bv
      type:
        scalar: boolean
    - name: nv
      type:
        scalar: numeric
- name: myValue
  map:
    elementType:
      scalar: string
`,
	quints: []symdiffQuint{{
		lhs:      `{}`,
		rhs:      `{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`,
		removed:  _NS(),
		modified: _NS(),
		added: _NS(
			_P("list"),
			_P("list", _KBF("key", "a", "id", 1)),
			_P("list", _KBF("key", "a", "id", 1), "key"),
			_P("list", _KBF("key", "a", "id", 1), "id"),
			_P("list", _KBF("key", "a", "id", 1), "value"),
			_P("list", _KBF("key", "a", "id", 1), "value", "a"),
		),
	}, {
		lhs:      `{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`,
		rhs:      `{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(),
	}, {
		lhs:      `{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`,
		rhs:      `{"list":[{"key":"a","id":1,"value":{"a":"b"}}]}`,
		removed:  _NS(),
		modified: _NS(_P("list", _KBF("key", "a", "id", 1), "value", "a")),
		added:    _NS(),
	}, {
		lhs: `{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`,
		rhs: `{"list":[{"key":"a","id":2,"value":{"a":"a"}}]}`,
		removed: _NS(
			_P("list", _KBF("key", "a", "id", 1)),
			_P("list", _KBF("key", "a", "id", 1), "key"),
			_P("list", _KBF("key", "a", "id", 1), "id"),
			_P("list", _KBF("key", "a", "id", 1), "value"),
			_P("list", _KBF("key", "a", "id", 1), "value", "a"),
		),
		modified: _NS(),
		added: _NS(
			_P("list", _KBF("key", "a", "id", 2)),
			_P("list", _KBF("key", "a", "id", 2), "key"),
			_P("list", _KBF("key", "a", "id", 2), "id"),
			_P("list", _KBF("key", "a", "id", 2), "value"),
			_P("list", _KBF("key", "a", "id", 2), "value", "a"),
		),
	}, {
		lhs: `{"list":[{"key":"a","id":1},{"key":"b","id":1}]}`,
		rhs: `{"list":[{"key":"a","id":1},{"key":"a","id":2}]}`,
		removed: _NS(
			_P("list", _KBF("key", "b", "id", 1)),
			_P("list", _KBF("key", "b", "id", 1), "key"),
			_P("list", _KBF("key", "b", "id", 1), "id"),
		),
		modified: _NS(),
		added: _NS(
			_P("list", _KBF("key", "a", "id", 2)),
			_P("list", _KBF("key", "a", "id", 2), "key"),
			_P("list", _KBF("key", "a", "id", 2), "id"),
		),
	}, {
		lhs:      `{"atomicList":["a","a","a"]}`,
		rhs:      `{"atomicList":null}`,
		removed:  _NS(),
		modified: _NS(_P("atomicList")),
		added:    _NS(),
	}, {
		lhs:      `{"atomicList":["a","b","c"]}`,
		rhs:      `{"atomicList":[]}`,
		removed:  _NS(),
		modified: _NS(_P("atomicList")),
		added:    _NS(),
	}, {
		lhs:      `{"atomicList":["a","a","a"]}`,
		rhs:      `{"atomicList":["a","a"]}`,
		removed:  _NS(),
		modified: _NS(_P("atomicList")),
		added:    _NS(),
	}},
}}

func (tt symdiffTestCase) test(t *testing.T) {
	parser, err := typed.NewParser(tt.schema)
	if err != nil {
		t.Fatalf("failed to create schema: %v", err)
	}
	for i, quint := range tt.quints {
		quint := quint
		t.Run(fmt.Sprintf("%v-valid-%v", tt.name, i), func(t *testing.T) {
			t.Parallel()
			pt := parser.Type(tt.rootTypeName)

			tvLHS, err := pt.FromYAML(quint.lhs)
			if err != nil {
				t.Fatalf("failed to parse lhs: %v", err)
			}
			tvRHS, err := pt.FromYAML(quint.rhs)
			if err != nil {
				t.Fatalf("failed to parse rhs: %v", err)
			}
			got, err := tvLHS.Compare(tvRHS)
			if err != nil {
				t.Fatalf("got validation errors: %v", err)
			}
			t.Logf("got added:\n%s\n", got.Added)
			if !got.Added.Equals(quint.added) {
				t.Errorf("Expected added:\n%s\n", quint.added)
			}
			t.Logf("got modified:\n%s", got.Modified)
			if !got.Modified.Equals(quint.modified) {
				t.Errorf("Expected modified:\n%s\n", quint.modified)
			}
			t.Logf("got removed:\n%s", got.Removed)
			if !got.Removed.Equals(quint.removed) {
				t.Errorf("Expected removed:\n%s\n", quint.removed)
			}

			// Do the reverse operation and sanity check.
			gotR, err := tvRHS.Compare(tvLHS)
			if err != nil {
				t.Fatalf("(reverse) got validation errors: %v", err)
			}
			if !gotR.Modified.Equals(got.Modified) {
				t.Errorf("reverse operation gave different modified list:\n%s", gotR.Modified)
			}
			if !gotR.Removed.Equals(got.Added) {
				t.Errorf("reverse removed gave different result than added:\n%s", gotR.Removed)
			}
			if !gotR.Added.Equals(got.Removed) {
				t.Errorf("reverse added gave different result than removed:\n%s", gotR.Added)
			}

		})
	}
}

func TestSymdiff(t *testing.T) {
	for _, tt := range symdiffCases {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			tt.test(t)
		})
	}
}
