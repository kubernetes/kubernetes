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

// Package encodingtest provides helpers for testing that types with both
// json/v1 and json/v2 methods encode identically through every JSON entry
// point, and that they work with YAML.
package encodingtest

import (
	"encoding/json"
	jsonv2 "encoding/json/v2"
	"reflect"
	"strconv"
	"testing"

	"sigs.k8s.io/yaml"

	utiljson "k8s.io/apimachinery/pkg/util/json"
)

// MarshalCase is a value together with its expected encoding.
type MarshalCase[T any] struct {
	Input  T
	Result string
}

// UnmarshalCase is an encoded value together with its expected decoding.
type UnmarshalCase[T any] struct {
	Input  string
	Result T
}

// variant is one way of encoding and decoding T.
type variant[T any] struct {
	marshal   func(T) ([]byte, error)
	unmarshal func([]byte) (T, error)
}

// legacy hides the json/v2 methods of T: it only has MarshalJSON and
// UnmarshalJSON, which call the corresponding methods of T.
//
// Accessing the underlying MarshalJSON and UnmarshalJSON through runtime
// casting is the simplest solution because some types only implement them for
// pointer receivers and there's no good way to declare that a pointer to T
// must implement the interfaces. All other solutions are worse.
type legacy[T any] struct{ v T }

func (l legacy[T]) MarshalJSON() ([]byte, error) {
	return any(&l.v).(json.Marshaler).MarshalJSON()
}

func (l *legacy[T]) UnmarshalJSON(b []byte) error {
	return any(&l.v).(json.Unmarshaler).UnmarshalJSON(b)
}

// variants returns the ways of encoding T that must behave identically:
// the legacy MarshalJSON/UnmarshalJSON methods, encoding/json,
// k8s.io/apimachinery/pkg/util/json and encoding/json/v2.
//
// Since Go 1.27, encoding/json prefers the json/v2 methods, so the legacy
// methods are only reached through legacy[T].
func variants[T any]() map[string]variant[T] {
	return map[string]variant[T]{
		"legacy-methods": {
			marshal: func(v T) ([]byte, error) { return json.Marshal(legacy[T]{v}) },
			unmarshal: func(b []byte) (T, error) {
				var l legacy[T]
				err := json.Unmarshal(b, &l)
				return l.v, err
			},
		},
		"json": {
			marshal: func(v T) ([]byte, error) { return json.Marshal(v) },
			unmarshal: func(b []byte) (T, error) {
				var v T
				err := json.Unmarshal(b, &v)
				return v, err
			},
		},
		"utiljson": {
			marshal: func(v T) ([]byte, error) { return utiljson.Marshal(v) },
			unmarshal: func(b []byte) (T, error) {
				var v T
				err := utiljson.Unmarshal(b, &v)
				return v, err
			},
		},
		"jsonv2": {
			marshal: func(v T) ([]byte, error) { return jsonv2.Marshal(v) },
			unmarshal: func(b []byte) (T, error) {
				var v T
				err := jsonv2.Unmarshal(b, &v)
				return v, err
			},
		},
	}
}

// MarshalJSON checks that all variants encode Input as Result.
func MarshalJSON[T any](t *testing.T, cases []MarshalCase[T]) {
	for name, variant := range variants[T]() {
		t.Run(name, func(t *testing.T) {
			for i, c := range cases {
				t.Run(strconv.Itoa(i), func(t *testing.T) {
					got, err := variant.marshal(c.Input)
					if err != nil {
						t.Errorf("failed to marshal input %#v: %v", c.Input, err)
					} else if string(got) != c.Result {
						t.Errorf("failed to marshal input %#v: expected %s, got %s", c.Input, c.Result, got)
					}
				})
			}
		})
	}
}

// UnmarshalJSON checks that all variants decode Input as Result.
func UnmarshalJSON[T any](t *testing.T, cases []UnmarshalCase[T]) {
	for name, variant := range variants[T]() {
		t.Run(name, func(t *testing.T) {
			for i, c := range cases {
				t.Run(strconv.Itoa(i), func(t *testing.T) {
					got, err := variant.unmarshal([]byte(c.Input))
					if err != nil {
						t.Errorf("failed to unmarshal input %s: %v", c.Input, err)
					} else if !reflect.DeepEqual(got, c.Result) {
						t.Errorf("failed to unmarshal input %s: expected %#v, got %#v", c.Input, c.Result, got)
					}
				})
			}
		})
	}
}

// MarshalJSONErrors checks that all variants fail to encode the inputs.
func MarshalJSONErrors[T any](t *testing.T, inputs []T) {
	for name, variant := range variants[T]() {
		t.Run(name+"-errors", func(t *testing.T) {
			for i, input := range inputs {
				t.Run(strconv.Itoa(i), func(t *testing.T) {
					if got, err := variant.marshal(input); err == nil {
						t.Errorf("expected error for input %#v, got %s", input, got)
					}
				})
			}
		})
	}
}

// UnmarshalJSONErrors checks that all variants fail to decode the inputs.
func UnmarshalJSONErrors[T any](t *testing.T, inputs []string) {
	for name, variant := range variants[T]() {
		t.Run(name+"-errors", func(t *testing.T) {
			for i, input := range inputs {
				t.Run(strconv.Itoa(i), func(t *testing.T) {
					if got, err := variant.unmarshal([]byte(input)); err == nil {
						t.Errorf("expected error for input %s, got %#v", input, got)
					}
				})
			}
		})
	}
}

// MarshalYAML checks that Input is encoded as Result.
func MarshalYAML[T any](t *testing.T, cases []MarshalCase[T]) {
	for i, c := range cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			got, err := yaml.Marshal(c.Input)
			if err != nil {
				t.Errorf("failed to marshal input %#v: %v", c.Input, err)
			} else if string(got) != c.Result {
				t.Errorf("failed to marshal input %#v: expected %q, got %q", c.Input, c.Result, got)
			}
		})
	}
}

// UnmarshalYAML checks that Input is decoded as Result.
func UnmarshalYAML[T any](t *testing.T, cases []UnmarshalCase[T]) {
	for i, c := range cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			var got T
			if err := yaml.Unmarshal([]byte(c.Input), &got); err != nil {
				t.Errorf("failed to unmarshal input %q: %v", c.Input, err)
			} else if !reflect.DeepEqual(got, c.Result) {
				t.Errorf("failed to unmarshal input %q: expected %#v, got %#v", c.Input, c.Result, got)
			}
		})
	}
}

// JSONToYAML checks that the JSON encoding of each input can be decoded as
// YAML into a value which satisfies equal. A nil equal compares with
// reflect.DeepEqual.
func JSONToYAML[T any](t *testing.T, inputs []T, equal func(a, b T) bool) {
	if equal == nil {
		equal = func(a, b T) bool { return reflect.DeepEqual(a, b) }
	}
	for i, input := range inputs {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			data, err := json.Marshal(input)
			if err != nil {
				t.Fatalf("failed to marshal input %#v: %v", input, err)
			}
			var got T
			if err := yaml.Unmarshal(data, &got); err != nil {
				t.Errorf("failed to unmarshal %s: %v", data, err)
			} else if !equal(got, input) {
				t.Errorf("JSON %s: expected %#v, got %#v", data, input, got)
			}
		})
	}
}

// BenchmarkMarshalJSON measures encoding of all inputs per iteration, once per
// variant.
func BenchmarkMarshalJSON[T any](b *testing.B, cases []MarshalCase[T]) {
	for name, variant := range variants[T]() {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				for _, c := range cases {
					if _, err := variant.marshal(c.Input); err != nil {
						b.Fatalf("failed to marshal input %#v: %v", c.Input, err)
					}
				}
			}
		})
	}
}

// BenchmarkUnmarshalJSON measures decoding of all inputs per iteration, once
// per variant.
func BenchmarkUnmarshalJSON[T any](b *testing.B, cases []UnmarshalCase[T]) {
	for name, variant := range variants[T]() {
		inputs := make([][]byte, len(cases))
		for i, c := range cases {
			inputs[i] = []byte(c.Input)
		}
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				for i, input := range inputs {
					if _, err := variant.unmarshal(input); err != nil {
						b.Fatalf("failed to unmarshal input %s: %v", cases[i].Input, err)
					}
				}
			}
		})
	}
}
