package influxql_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/pkg/deep"
)

func TestPoint_Clone_Float(t *testing.T) {
	p := &influxql.FloatPoint{
		Name:  "cpu",
		Tags:  ParseTags("host=server01"),
		Time:  5,
		Value: 2,
		Aux:   []interface{}{float64(45)},
	}
	c := p.Clone()
	if p == c {
		t.Errorf("clone has the same address as the original: %v == %v", p, c)
	}
	if !deep.Equal(p, c) {
		t.Errorf("mismatched point: %s", spew.Sdump(c))
	}
	if &p.Aux[0] == &c.Aux[0] {
		t.Errorf("aux values share the same address: %v == %v", p.Aux, c.Aux)
	} else if !deep.Equal(p.Aux, c.Aux) {
		t.Errorf("mismatched aux fields: %v != %v", p.Aux, c.Aux)
	}
}

func TestPoint_Clone_Integer(t *testing.T) {
	p := &influxql.IntegerPoint{
		Name:  "cpu",
		Tags:  ParseTags("host=server01"),
		Time:  5,
		Value: 2,
		Aux:   []interface{}{float64(45)},
	}
	c := p.Clone()
	if p == c {
		t.Errorf("clone has the same address as the original: %v == %v", p, c)
	}
	if !deep.Equal(p, c) {
		t.Errorf("mismatched point: %s", spew.Sdump(c))
	}
	if &p.Aux[0] == &c.Aux[0] {
		t.Errorf("aux values share the same address: %v == %v", p.Aux, c.Aux)
	} else if !deep.Equal(p.Aux, c.Aux) {
		t.Errorf("mismatched aux fields: %v != %v", p.Aux, c.Aux)
	}
}

func TestPoint_Clone_String(t *testing.T) {
	p := &influxql.StringPoint{
		Name:  "cpu",
		Tags:  ParseTags("host=server01"),
		Time:  5,
		Value: "clone",
		Aux:   []interface{}{float64(45)},
	}
	c := p.Clone()
	if p == c {
		t.Errorf("clone has the same address as the original: %v == %v", p, c)
	}
	if !deep.Equal(p, c) {
		t.Errorf("mismatched point: %s", spew.Sdump(c))
	}
	if &p.Aux[0] == &c.Aux[0] {
		t.Errorf("aux values share the same address: %v == %v", p.Aux, c.Aux)
	} else if !deep.Equal(p.Aux, c.Aux) {
		t.Errorf("mismatched aux fields: %v != %v", p.Aux, c.Aux)
	}
}

func TestPoint_Clone_Boolean(t *testing.T) {
	p := &influxql.BooleanPoint{
		Name:  "cpu",
		Tags:  ParseTags("host=server01"),
		Time:  5,
		Value: true,
		Aux:   []interface{}{float64(45)},
	}
	c := p.Clone()
	if p == c {
		t.Errorf("clone has the same address as the original: %v == %v", p, c)
	}
	if !deep.Equal(p, c) {
		t.Errorf("mismatched point: %s", spew.Sdump(c))
	}
	if &p.Aux[0] == &c.Aux[0] {
		t.Errorf("aux values share the same address: %v == %v", p.Aux, c.Aux)
	} else if !deep.Equal(p.Aux, c.Aux) {
		t.Errorf("mismatched aux fields: %v != %v", p.Aux, c.Aux)
	}
}

func TestPoint_Clone_Nil(t *testing.T) {
	var fp *influxql.FloatPoint
	if p := fp.Clone(); p != nil {
		t.Errorf("expected nil, got %v", p)
	}

	var ip *influxql.IntegerPoint
	if p := ip.Clone(); p != nil {
		t.Errorf("expected nil, got %v", p)
	}

	var sp *influxql.StringPoint
	if p := sp.Clone(); p != nil {
		t.Errorf("expected nil, got %v", p)
	}

	var bp *influxql.BooleanPoint
	if p := bp.Clone(); p != nil {
		t.Errorf("expected nil, got %v", p)
	}
}

// TestPoint_Fields ensures that no additional fields are added to the point structs.
// This struct is very sensitive and can effect performance unless handled carefully.
// To avoid the struct becoming a dumping ground for every function that needs to store
// miscellaneous information, this test is meant to ensure that new fields don't slip
// into the struct.
func TestPoint_Fields(t *testing.T) {
	allowedFields := map[string]bool{
		"Name":       true,
		"Tags":       true,
		"Time":       true,
		"Nil":        true,
		"Value":      true,
		"Aux":        true,
		"Aggregated": true,
	}

	for _, typ := range []reflect.Type{
		reflect.TypeOf(influxql.FloatPoint{}),
		reflect.TypeOf(influxql.IntegerPoint{}),
		reflect.TypeOf(influxql.StringPoint{}),
		reflect.TypeOf(influxql.BooleanPoint{}),
	} {
		f, ok := typ.FieldByNameFunc(func(name string) bool {
			return !allowedFields[name]
		})
		if ok {
			t.Errorf("found an unallowed field in %s: %s %s", typ, f.Name, f.Type)
		}
	}
}

// Ensure that tags can return a unique id.
func TestTags_ID(t *testing.T) {
	tags := influxql.NewTags(map[string]string{"foo": "bar", "baz": "bat"})
	if id := tags.ID(); id != "baz\x00foo\x00bat\x00bar" {
		t.Fatalf("unexpected id: %q", id)
	}
}

// Ensure that a subset can be created from a tag set.
func TestTags_Subset(t *testing.T) {
	tags := influxql.NewTags(map[string]string{"a": "0", "b": "1", "c": "2"})
	subset := tags.Subset([]string{"b", "c", "d"})
	if keys := subset.Keys(); !reflect.DeepEqual(keys, []string{"b", "c", "d"}) {
		t.Fatalf("unexpected keys: %+v", keys)
	} else if v := subset.Value("a"); v != "" {
		t.Fatalf("unexpected 'a' value: %s", v)
	} else if v := subset.Value("b"); v != "1" {
		t.Fatalf("unexpected 'b' value: %s", v)
	} else if v := subset.Value("c"); v != "2" {
		t.Fatalf("unexpected 'c' value: %s", v)
	} else if v := subset.Value("d"); v != "" {
		t.Fatalf("unexpected 'd' value: %s", v)
	}
}

// ParseTags returns an instance of Tags for a comma-delimited list of key/values.
func ParseTags(s string) influxql.Tags {
	m := make(map[string]string)
	for _, kv := range strings.Split(s, ",") {
		a := strings.Split(kv, "=")
		m[a[0]] = a[1]
	}
	return influxql.NewTags(m)
}
