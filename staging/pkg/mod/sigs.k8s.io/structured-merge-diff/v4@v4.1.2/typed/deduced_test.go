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
	"fmt"
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/typed"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

func TestValidateDeducedType(t *testing.T) {
	tests := []string{
		`{"a": null}`,
		`{"a": ["a", "b"]}`,
		`{"a": {"b": [], "c": 2, "d": {"f": "string"}}}`,
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("test %d", i), func(t *testing.T) {
			v, err := typed.DeducedParseableType.FromYAML(typed.YAMLObject(test))
			if err != nil {
				t.Fatalf("Failed to parse yaml: %v", err)
			}
			if err := v.Validate(); err != nil {
				t.Fatalf("Validation failed: %v", err)
			}
		})
	}
}

func TestMergeDeduced(t *testing.T) {
	triplets := []mergeTriplet{
		{
			`{"key":"foo","value":{}}`,
			`{"key":"foo","value":1}`,
			`{"key":"foo","value":1}`,
		}, {
			`{"key":"foo","value":1}`,
			`{"key":"foo","value":{}}`,
			`{"key":"foo","value":{}}`,
		}, {
			`{"key":"foo","value":null}`,
			`{"key":"foo","value":{}}`,
			`{"key":"foo","value":{}}`,
		}, {
			`{"key":"foo"}`,
			`{"value":true}`,
			`{"key":"foo","value":true}`,
		}, {
			`{}`,
			`{"inner":{}}`,
			`{"inner":{}}`,
		}, {
			`{}`,
			`{"inner":null}`,
			`{"inner":null}`,
		}, {
			`{"inner":null}`,
			`{"inner":{}}`,
			`{"inner":{}}`,
		}, {
			`{"inner":{}}`,
			`{"inner":null}`,
			`{"inner":null}`,
		}, {
			`{"inner":{}}`,
			`{"inner":{}}`,
			`{"inner":{}}`,
		}, {
			`{}`,
			`{"inner":{}}`,
			`{"inner":{}}`,
		}, {
			`{"inner":null}`,
			`{"inner":{}}`,
			`{"inner":{}}`,
		}, {
			`{"inner":{}}`,
			`{"inner":null}`,
			`{"inner":null}`,
		}, {
			`{}`,
			`{"inner":[]}`,
			`{"inner":[]}`,
		}, {
			`{"inner":null}`,
			`{"inner":[]}`,
			`{"inner":[]}`,
		}, {
			`{"inner":[]}`,
			`{"inner":null}`,
			`{"inner":null}`,
		}, {
			`{"inner":[]}`,
			`{"inner":[]}`,
			`{"inner":[]}`,
		}, {
			`{"numeric":1}`,
			`{"numeric":3.14159}`,
			`{"numeric":3.14159}`,
		}, {
			`{"numeric":3.14159}`,
			`{"numeric":1}`,
			`{"numeric":1}`,
		}, {
			`{"string":"aoeu"}`,
			`{"bool":true}`,
			`{"string":"aoeu","bool":true}`,
		}, {
			`{"atomic":["a","b","c"]}`,
			`{"atomic":["a","b"]}`,
			`{"atomic":["a","b"]}`,
		}, {
			`{"atomic":["a","b"]}`,
			`{"atomic":["a","b","c"]}`,
			`{"atomic":["a","b","c"]}`,
		}, {
			`{"atomic":["a","b","c"]}`,
			`{"atomic":[]}`,
			`{"atomic":[]}`,
		}, {
			`{"atomic":[]}`,
			`{"atomic":["a","b","c"]}`,
			`{"atomic":["a","b","c"]}`,
		}, {
			`{"":[true]}`,
			`{"setBool":[false]}`,
			`{"":[true],"setBool":[false]}`,
		}, {
			`{"atomic":[1,2,3.14159]}`,
			`{"atomic":[1,2,3]}`,
			`{"atomic":[1,2,3]}`,
		}, {
			`{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`,
			`{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`,
			`{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`,
		}, {
			`{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`,
			`{"list":[{"key":"a","id":2,"value":{"a":"a"}}]}`,
			`{"list":[{"key":"a","id":2,"value":{"a":"a"}}]}`,
		}, {
			`{"list":[{"key":"a","id":1},{"key":"b","id":1}]}`,
			`{"list":[{"key":"a","id":1},{"key":"a","id":2}]}`,
			`{"list":[{"key":"a","id":1},{"key":"a","id":2}]}`,
		}, {
			`{"atomicList":["a","a","a"]}`,
			`{"atomicList":null}`,
			`{"atomicList":null}`,
		}, {
			`{"atomicList":["a","b","c"]}`,
			`{"atomicList":[]}`,
			`{"atomicList":[]}`,
		}, {
			`{"atomicList":["a","a","a"]}`,
			`{"atomicList":["a","a"]}`,
			`{"atomicList":["a","a"]}`,
		}, {
			`{"a":1,"b":[null],"c":{"id":2,"list":["value"]}}`,
			`{"a":2,"b":["value"],"c":{"name":"my_name"}}`,
			`{"a":2,"b":["value"],"c":{"id":2,"list":["value"],"name":"my_name"}}`,
		}}

	for i, triplet := range triplets {
		triplet := triplet
		t.Run(fmt.Sprintf("triplet-%v", i), func(t *testing.T) {
			t.Parallel()

			pt := typed.DeducedParseableType
			lhs, err := pt.FromYAML(triplet.lhs)
			if err != nil {
				t.Fatalf("unable to parser/validate lhs yaml: %v\n%v", err, triplet.lhs)
			}

			rhs, err := pt.FromYAML(triplet.rhs)
			if err != nil {
				t.Fatalf("unable to parser/validate rhs yaml: %v\n%v", err, triplet.rhs)
			}

			out, err := pt.FromYAML(triplet.out)
			if err != nil {
				t.Fatalf("unable to parser/validate out yaml: %v\n%v", err, triplet.out)
			}

			got, err := lhs.Merge(rhs)
			if err != nil {
				t.Errorf("got validation errors: %v", err)
			} else {
				if !value.Equals(got.AsValue(), out.AsValue()) {
					t.Errorf("Expected\n%v\nbut got\n%v\n",
						value.ToString(out.AsValue()), value.ToString(got.AsValue()),
					)
				}
			}
		})
	}
}

func TestToSetDeduced(t *testing.T) {
	tests := []objSetPair{
		{`{"key":"foo","value":1}`, _NS(_P("key"), _P("value"))},
		{`{"key":"foo","value":{"a": "b"}}`, _NS(_P("key"), _P("value"), _P("value", "a"))},
		{`{"key":"foo","value":null}`, _NS(_P("key"), _P("value"))},
		{`{"key":"foo"}`, _NS(_P("key"))},
		{`{"key":"foo","value":true}`, _NS(_P("key"), _P("value"))},
		{`{"numeric":1}`, _NS(_P("numeric"))},
		{`{"numeric":3.14159}`, _NS(_P("numeric"))},
		{`{"string":"aoeu"}`, _NS(_P("string"))},
		{`{"bool":true}`, _NS(_P("bool"))},
		{`{"bool":false}`, _NS(_P("bool"))},
		{`{"list":["a","b","c"]}`, _NS(_P("list"))},
		{`{"color":{}}`, _NS(_P("color"))},
		{`{"color":null}`, _NS(_P("color"))},
		{`{"color":{"R":255,"G":0,"B":0}}`, _NS(_P("color"), _P("color", "R"), _P("color", "G"), _P("color", "B"))},
		{`{"arbitraryWavelengthColor":null}`, _NS(_P("arbitraryWavelengthColor"))},
		{`{"arbitraryWavelengthColor":{"IR":255}}`, _NS(_P("arbitraryWavelengthColor"), _P("arbitraryWavelengthColor", "IR"))},
		{`{"args":[]}`, _NS(_P("args"))},
		{`{"args":null}`, _NS(_P("args"))},
		{`{"args":[null]}`, _NS(_P("args"))},
		{`{"args":[{"key":"a","value":"b"},{"key":"c","value":"d"}]}`, _NS(_P("args"))},
		{`{"atomicList":["a","a","a"]}`, _NS(_P("atomicList"))},
	}

	for i, v := range tests {
		v := v
		t.Run(fmt.Sprintf("%v", i), func(t *testing.T) {
			t.Parallel()

			tv, err := typed.DeducedParseableType.FromYAML(v.object)
			if err != nil {
				t.Fatalf("failed to parse object: %v", err)
			}
			fs, err := tv.ToFieldSet()
			if err != nil {
				t.Fatalf("got validation errors: %v", err)
			}
			if !fs.Equals(v.set) {
				t.Errorf("wanted\n%s\ngot\n%s\n", v.set, fs)
			}
		})
	}
}

func TestSymdiffDeduced(t *testing.T) {
	quints := []symdiffQuint{{
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
		rhs:      `{"key":"foo","value":{"deep":{"nested":1}}}`,
		removed:  _NS(),
		modified: _NS(_P("value")),
		added:    _NS(_P("value", "deep"), _P("value", "deep", "nested")),
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
	}, {
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
	}, {
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
	}, {
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
		lhs:      `{"a":{"b":{"c":[true]}}}`,
		rhs:      `{"a":{"b":[false]}}`,
		removed:  _NS(_P("a", "b", "c")),
		modified: _NS(_P("a", "b")),
		added:    _NS(),
	}, {
		lhs:      `{"a":{}}`,
		rhs:      `{"a":{"b":"true"}}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("a", "b")),
	}, {
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
		lhs:      `{"list":["a","b"]}`,
		rhs:      `{"list":["a","b","c"]}`,
		removed:  _NS(),
		modified: _NS(_P("list")),
		added:    _NS(),
	}, {
		lhs:      `{}`,
		rhs:      `{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`,
		removed:  _NS(),
		modified: _NS(),
		added:    _NS(_P("list")),
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
		modified: _NS(_P("list")),
		added:    _NS(),
	}, {
		lhs:      `{"atomicList":["a","a","a"]}`,
		rhs:      `{"atomicList":null}`,
		removed:  _NS(),
		modified: _NS(_P("atomicList")),
		added:    _NS(),
	}, {
		lhs:      `{"atomicList":["a","a","a"]}`,
		rhs:      `{"atomicList":["a","a"]}`,
		removed:  _NS(),
		modified: _NS(_P("atomicList")),
		added:    _NS(),
	}}

	for i, quint := range quints {
		quint := quint
		t.Run(fmt.Sprintf("%v", i), func(t *testing.T) {
			//t.Parallel()
			pt := typed.DeducedParseableType

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
