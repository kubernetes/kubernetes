package misc_tests

import (
	"encoding/json"
	"github.com/json-iterator/go"
	"reflect"
	"strings"
	"testing"
)

type Level1 struct {
	Hello []Level2
}

type Level2 struct {
	World string
}

func Test_deep_nested(t *testing.T) {
	type unstructured interface{}

	testcases := []struct {
		name        string
		data        []byte
		expectError string
	}{
		{
			name:        "array under maxDepth",
			data:        []byte(`{"a":` + strings.Repeat(`[`, 10000-1) + strings.Repeat(`]`, 10000-1) + `}`),
			expectError: "",
		},
		{
			name:        "array over maxDepth",
			data:        []byte(`{"a":` + strings.Repeat(`[`, 10000) + strings.Repeat(`]`, 10000) + `}`),
			expectError: "max depth",
		},
		{
			name:        "object under maxDepth",
			data:        []byte(`{"a":` + strings.Repeat(`{"a":`, 10000-1) + `0` + strings.Repeat(`}`, 10000-1) + `}`),
			expectError: "",
		},
		{
			name:        "object over maxDepth",
			data:        []byte(`{"a":` + strings.Repeat(`{"a":`, 10000) + `0` + strings.Repeat(`}`, 10000) + `}`),
			expectError: "max depth",
		},
	}

	targets := []struct {
		name string
		new  func() interface{}
	}{
		{
			name: "unstructured",
			new: func() interface{} {
				var v interface{}
				return &v
			},
		},
		{
			name: "typed named field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
				}{}
				return &v
			},
		},
		{
			name: "typed missing field",
			new: func() interface{} {
				v := struct {
					B interface{} `json:"b"`
				}{}
				return &v
			},
		},
		{
			name: "typed 1 field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
				}{}
				return &v
			},
		},
		{
			name: "typed 2 field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
					B interface{} `json:"b"`
				}{}
				return &v
			},
		},
		{
			name: "typed 3 field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
					B interface{} `json:"b"`
					C interface{} `json:"c"`
				}{}
				return &v
			},
		},
		{
			name: "typed 4 field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
					B interface{} `json:"b"`
					C interface{} `json:"c"`
					D interface{} `json:"d"`
				}{}
				return &v
			},
		},
		{
			name: "typed 5 field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
					B interface{} `json:"b"`
					C interface{} `json:"c"`
					D interface{} `json:"d"`
					E interface{} `json:"e"`
				}{}
				return &v
			},
		},
		{
			name: "typed 6 field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
					B interface{} `json:"b"`
					C interface{} `json:"c"`
					D interface{} `json:"d"`
					E interface{} `json:"e"`
					F interface{} `json:"f"`
				}{}
				return &v
			},
		},
		{
			name: "typed 7 field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
					B interface{} `json:"b"`
					C interface{} `json:"c"`
					D interface{} `json:"d"`
					E interface{} `json:"e"`
					F interface{} `json:"f"`
					G interface{} `json:"g"`
				}{}
				return &v
			},
		},
		{
			name: "typed 8 field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
					B interface{} `json:"b"`
					C interface{} `json:"c"`
					D interface{} `json:"d"`
					E interface{} `json:"e"`
					F interface{} `json:"f"`
					G interface{} `json:"g"`
					H interface{} `json:"h"`
				}{}
				return &v
			},
		},
		{
			name: "typed 9 field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
					B interface{} `json:"b"`
					C interface{} `json:"c"`
					D interface{} `json:"d"`
					E interface{} `json:"e"`
					F interface{} `json:"f"`
					G interface{} `json:"g"`
					H interface{} `json:"h"`
					I interface{} `json:"i"`
				}{}
				return &v
			},
		},
		{
			name: "typed 10 field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
					B interface{} `json:"b"`
					C interface{} `json:"c"`
					D interface{} `json:"d"`
					E interface{} `json:"e"`
					F interface{} `json:"f"`
					G interface{} `json:"g"`
					H interface{} `json:"h"`
					I interface{} `json:"i"`
					J interface{} `json:"j"`
				}{}
				return &v
			},
		},
		{
			name: "typed 11 field",
			new: func() interface{} {
				v := struct {
					A interface{} `json:"a"`
					B interface{} `json:"b"`
					C interface{} `json:"c"`
					D interface{} `json:"d"`
					E interface{} `json:"e"`
					F interface{} `json:"f"`
					G interface{} `json:"g"`
					H interface{} `json:"h"`
					I interface{} `json:"i"`
					J interface{} `json:"j"`
					K interface{} `json:"k"`
				}{}
				return &v
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			for _, target := range targets {
				t.Run(target.name, func(t *testing.T) {
					err := jsoniter.Unmarshal(tc.data, target.new())
					if len(tc.expectError) == 0 {
						if err != nil {
							t.Errorf("unexpected error: %v", err)
						}
					} else {
						if err == nil {
							t.Errorf("expected error, got none")
						} else if !strings.Contains(err.Error(), tc.expectError) {
							t.Errorf("expected error containing '%s', got: %v", tc.expectError, err)
						}
					}
				})
			}
		})
	}
}

func Test_nested(t *testing.T) {
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, `{"hello": [{"world": "value1"}, {"world": "value2"}]}`)
	l1 := Level1{}
	for l1Field := iter.ReadObject(); l1Field != ""; l1Field = iter.ReadObject() {
		switch l1Field {
		case "hello":
			l2Array := []Level2{}
			for iter.ReadArray() {
				l2 := Level2{}
				for l2Field := iter.ReadObject(); l2Field != ""; l2Field = iter.ReadObject() {
					switch l2Field {
					case "world":
						l2.World = iter.ReadString()
					default:
						iter.ReportError("bind l2", "unexpected field: "+l2Field)
					}
				}
				l2Array = append(l2Array, l2)
			}
			l1.Hello = l2Array
		default:
			iter.ReportError("bind l1", "unexpected field: "+l1Field)
		}
	}
	if !reflect.DeepEqual(l1, Level1{
		Hello: []Level2{
			{World: "value1"},
			{World: "value2"},
		},
	}) {
		t.Fatal(l1)
	}
}

func Benchmark_jsoniter_nested(b *testing.B) {
	for n := 0; n < b.N; n++ {
		iter := jsoniter.ParseString(jsoniter.ConfigDefault, `{"hello": [{"world": "value1"}, {"world": "value2"}]}`)
		l1 := Level1{}
		for l1Field := iter.ReadObject(); l1Field != ""; l1Field = iter.ReadObject() {
			switch l1Field {
			case "hello":
				l1.Hello = readLevel1Hello(iter)
			default:
				iter.Skip()
			}
		}
	}
}

func readLevel1Hello(iter *jsoniter.Iterator) []Level2 {
	l2Array := make([]Level2, 0, 2)
	for iter.ReadArray() {
		l2 := Level2{}
		for l2Field := iter.ReadObject(); l2Field != ""; l2Field = iter.ReadObject() {
			switch l2Field {
			case "world":
				l2.World = iter.ReadString()
			default:
				iter.Skip()
			}
		}
		l2Array = append(l2Array, l2)
	}
	return l2Array
}

func Benchmark_json_nested(b *testing.B) {
	for n := 0; n < b.N; n++ {
		l1 := Level1{}
		json.Unmarshal([]byte(`{"hello": [{"world": "value1"}, {"world": "value2"}]}`), &l1)
	}
}
