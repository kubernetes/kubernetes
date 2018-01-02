package jsoniter

import (
	"encoding/json"
	"reflect"
	"testing"
)

type Level1 struct {
	Hello []Level2
}

type Level2 struct {
	World string
}

func Test_nested(t *testing.T) {
	iter := ParseString(ConfigDefault, `{"hello": [{"world": "value1"}, {"world": "value2"}]}`)
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
		iter := ParseString(ConfigDefault, `{"hello": [{"world": "value1"}, {"world": "value2"}]}`)
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

func readLevel1Hello(iter *Iterator) []Level2 {
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
