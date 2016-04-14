package stringreplace

import (
	"fmt"
	"reflect"
	"testing"
)

type sampleInnerStruct struct {
	Name   string
	Number int
	List   []string
	Map    map[string]string
}

type sampleStruct struct {
	Name         string
	Inner        sampleInnerStruct
	Ptr          *sampleInnerStruct
	MapInMap     map[string]map[string]string
	ArrayInArray [][]string
	Array        []string
	ArrayInMap   map[string][]interface{}
}

func TestVisitObjectStringsOnStruct(t *testing.T) {
	samples := [][]sampleStruct{
		{{}, {}},
		{{Name: "Foo"}, {Name: "sample-Foo"}},
		{{Ptr: nil}, {Ptr: nil}},
		{{Ptr: &sampleInnerStruct{Name: "foo"}}, {Ptr: &sampleInnerStruct{Name: "sample-foo"}}},
		{{Inner: sampleInnerStruct{Name: "foo"}}, {Inner: sampleInnerStruct{Name: "sample-foo"}}},
		{{Array: []string{"foo", "bar"}}, {Array: []string{"sample-foo", "sample-bar"}}},
		{
			{
				MapInMap: map[string]map[string]string{
					"foo": {"bar": "test"},
				},
			},
			{
				MapInMap: map[string]map[string]string{
					"foo": {"bar": "sample-test"},
				},
			},
		},
		{
			{ArrayInArray: [][]string{{"foo", "bar"}}},
			{ArrayInArray: [][]string{{"sample-foo", "sample-bar"}}},
		},
		{
			{ArrayInMap: map[string][]interface{}{"key": {"foo", "bar"}}},
			{ArrayInMap: map[string][]interface{}{"key": {"sample-foo", "sample-bar"}}},
		},
	}
	for i := range samples {
		VisitObjectStrings(&samples[i][0], func(in string) string {
			if len(in) == 0 {
				return in
			}
			return fmt.Sprintf("sample-%s", in)
		})
		if !reflect.DeepEqual(samples[i][0], samples[i][1]) {
			t.Errorf("Got %#v, expected %#v", samples[i][0], samples[i][1])
		}
	}
}

func TestVisitObjectStringsOnMap(t *testing.T) {
	samples := [][]map[string]string{
		{
			{"foo": "bar"},
			{"foo": "sample-bar"},
		},
		{
			{"empty": ""},
			{"empty": "sample-"},
		},
		{
			{"": "invalid"},
			{"": "sample-invalid"},
		},
	}

	for i := range samples {
		VisitObjectStrings(&samples[i][0], func(in string) string {
			return fmt.Sprintf("sample-%s", in)
		})
		if !reflect.DeepEqual(samples[i][0], samples[i][1]) {
			t.Errorf("Got %#v, expected %#v", samples[i][0], samples[i][1])
		}
	}
}

func TestVisitObjectStringsOnArray(t *testing.T) {
	samples := [][][]string{
		{
			{"foo", "bar"},
			{"sample-foo", "sample-bar"},
		},
	}

	for i := range samples {
		VisitObjectStrings(&samples[i][0], func(in string) string {
			return fmt.Sprintf("sample-%s", in)
		})
		if !reflect.DeepEqual(samples[i][0], samples[i][1]) {
			t.Errorf("Got %#v, expected %#v", samples[i][0], samples[i][1])
		}
	}
}
