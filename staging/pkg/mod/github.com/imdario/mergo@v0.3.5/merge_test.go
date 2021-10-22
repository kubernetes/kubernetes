package mergo

import (
	"reflect"
	"testing"
)

type transformer struct {
	m map[reflect.Type]func(dst, src reflect.Value) error
}

func (s *transformer) Transformer(t reflect.Type) func(dst, src reflect.Value) error {
	if fn, ok := s.m[t]; ok {
		return fn
	}
	return nil
}

type foo struct {
	s   string
	Bar *bar
}

type bar struct {
	i int
	s map[string]string
}

func TestMergeWithTransformerNilStruct(t *testing.T) {
	a := foo{s: "foo"}
	b := foo{Bar: &bar{i: 2, s: map[string]string{"foo": "bar"}}}
	if err := Merge(&a, &b, WithOverride, WithTransformers(&transformer{
		m: map[reflect.Type]func(dst, src reflect.Value) error{
			reflect.TypeOf(&bar{}): func(dst, src reflect.Value) error {
				// Do sthg with Elem
				t.Log(dst.Elem().FieldByName("i"))
				t.Log(src.Elem())
				return nil
			},
		},
	})); err != nil {
		t.Fatal(err)
	}
	if a.s != "foo" {
		t.Fatalf("b not merged in properly: a.s.Value(%s) != expected(%s)", a.s, "foo")
	}
	if a.Bar == nil {
		t.Fatalf("b not merged in properly: a.Bar shouldn't be nil")
	}
}
