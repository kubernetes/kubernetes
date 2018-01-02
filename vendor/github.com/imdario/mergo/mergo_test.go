// Copyright 2013 Dario Castañé. All rights reserved.
// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mergo

import (
	"gopkg.in/yaml.v1"
	"io/ioutil"
	"reflect"
	"testing"
)

type simpleTest struct {
	Value int
}

type complexTest struct {
	St simpleTest
	sz int
	Id string
}

type moreComplextText struct {
	Ct complexTest
	St simpleTest
	Nt simpleTest
}

type pointerTest struct {
	C *simpleTest
}

type sliceTest struct {
	S []int
}

func TestNil(t *testing.T) {
	if err := Merge(nil, nil); err != ErrNilArguments {
		t.Fail()
	}
}

func TestDifferentTypes(t *testing.T) {
	a := simpleTest{42}
	b := 42
	if err := Merge(&a, b); err != ErrDifferentArgumentsTypes {
		t.Fail()
	}
}

func TestSimpleStruct(t *testing.T) {
	a := simpleTest{}
	b := simpleTest{42}
	if err := Merge(&a, b); err != nil {
		t.FailNow()
	}
	if a.Value != 42 {
		t.Fatalf("b not merged in a properly: a.Value(%d) != b.Value(%d)", a.Value, b.Value)
	}
	if !reflect.DeepEqual(a, b) {
		t.FailNow()
	}
}

func TestComplexStruct(t *testing.T) {
	a := complexTest{}
	a.Id = "athing"
	b := complexTest{simpleTest{42}, 1, "bthing"}
	if err := Merge(&a, b); err != nil {
		t.FailNow()
	}
	if a.St.Value != 42 {
		t.Fatalf("b not merged in a properly: a.St.Value(%d) != b.St.Value(%d)", a.St.Value, b.St.Value)
	}
	if a.sz == 1 {
		t.Fatalf("a's private field sz not preserved from merge: a.sz(%d) == b.sz(%d)", a.sz, b.sz)
	}
	if a.Id != b.Id {
		t.Fatalf("a's field Id not merged properly: a.Id(%s) != b.Id(%s)", a.Id, b.Id)
	}
}

func TestPointerStruct(t *testing.T) {
	s1 := simpleTest{}
	s2 := simpleTest{19}
	a := pointerTest{&s1}
	b := pointerTest{&s2}
	if err := Merge(&a, b); err != nil {
		t.FailNow()
	}
	if a.C.Value != b.C.Value {
		//t.Fatalf("b not merged in a properly: a.C.Value(%d) != b.C.Value(%d)", a.C.Value, b.C.Value)
	}
}

func TestPointerStructNil(t *testing.T) {
	a := pointerTest{nil}
	b := pointerTest{&simpleTest{19}}
	if err := Merge(&a, b); err != nil {
		t.FailNow()
	}
	if a.C.Value != b.C.Value {
		t.Fatalf("b not merged in a properly: a.C.Value(%d) != b.C.Value(%d)", a.C.Value, b.C.Value)
	}
}

func TestSliceStruct(t *testing.T) {
	a := sliceTest{}
	b := sliceTest{[]int{1, 2, 3}}
	if err := Merge(&a, b); err != nil {
		t.FailNow()
	}
	if len(b.S) != 3 {
		t.FailNow()
	}
	if len(a.S) != len(b.S) {
		t.Fatalf("b not merged in a properly %d != %d", len(a.S), len(b.S))
	}

	a = sliceTest{[]int{1}}
	b = sliceTest{[]int{1, 2, 3}}
	if err := Merge(&a, b); err != nil {
		t.FailNow()
	}
	if len(b.S) != 3 {
		t.FailNow()
	}
	if len(a.S) != len(b.S) {
		t.Fatalf("b not merged in a properly %d != %d", len(a.S), len(b.S))
	}
}

func TestMaps(t *testing.T) {
	m := map[string]simpleTest{
		"a": simpleTest{},
		"b": simpleTest{42},
	}
	n := map[string]simpleTest{
		"a": simpleTest{16},
		"b": simpleTest{},
		"c": simpleTest{12},
	}
	if err := Merge(&m, n); err != nil {
		t.Fatalf(err.Error())
	}
	if len(m) != 3 {
		t.Fatalf(`n not merged in m properly, m must have 3 elements instead of %d`, len(m))
	}
	if m["a"].Value != 0 {
		t.Fatalf(`n merged in m because I solved non-addressable map values TODO: m["a"].Value(%d) != n["a"].Value(%d)`, m["a"].Value, n["a"].Value)
	}
	if m["b"].Value != 42 {
		t.Fatalf(`n wrongly merged in m: m["b"].Value(%d) != n["b"].Value(%d)`, m["b"].Value, n["b"].Value)
	}
	if m["c"].Value != 12 {
		t.Fatalf(`n not merged in m: m["c"].Value(%d) != n["c"].Value(%d)`, m["c"].Value, n["c"].Value)
	}
}

func TestYAMLMaps(t *testing.T) {
	thing := loadYAML("testdata/thing.yml")
	license := loadYAML("testdata/license.yml")
	ft := thing["fields"].(map[interface{}]interface{})
	fl := license["fields"].(map[interface{}]interface{})
	expectedLength := len(ft) + len(fl)
	if err := Merge(&license, thing); err != nil {
		t.Fatal(err.Error())
	}
	currentLength := len(license["fields"].(map[interface{}]interface{}))
	if currentLength != expectedLength {
		t.Fatalf(`thing not merged in license properly, license must have %d elements instead of %d`, expectedLength, currentLength)
	}
	fields := license["fields"].(map[interface{}]interface{})
	if _, ok := fields["id"]; !ok {
		t.Fatalf(`thing not merged in license properly, license must have a new id field from thing`)
	}
}

func TestTwoPointerValues(t *testing.T) {
	a := &simpleTest{}
	b := &simpleTest{42}
	if err := Merge(a, b); err != nil {
		t.Fatalf(`Boom. You crossed the streams: %s`, err)
	}
}

func TestMap(t *testing.T) {
	a := complexTest{}
	a.Id = "athing"
	c := moreComplextText{a, simpleTest{}, simpleTest{}}
	b := map[string]interface{}{
		"ct": map[string]interface{}{
			"st": map[string]interface{}{
				"value": 42,
			},
			"sz": 1,
			"id": "bthing",
		},
		"st": &simpleTest{144}, // Mapping a reference
		"zt": simpleTest{299},  // Mapping a missing field (zt doesn't exist)
		"nt": simpleTest{3},
	}
	if err := Map(&c, b); err != nil {
		t.FailNow()
	}
	m := b["ct"].(map[string]interface{})
	n := m["st"].(map[string]interface{})
	o := b["st"].(*simpleTest)
	p := b["nt"].(simpleTest)
	if c.Ct.St.Value != 42 {
		t.Fatalf("b not merged in a properly: c.Ct.St.Value(%d) != b.Ct.St.Value(%d)", c.Ct.St.Value, n["value"])
	}
	if c.St.Value != 144 {
		t.Fatalf("b not merged in a properly: c.St.Value(%d) != b.St.Value(%d)", c.St.Value, o.Value)
	}
	if c.Nt.Value != 3 {
		t.Fatalf("b not merged in a properly: c.Nt.Value(%d) != b.Nt.Value(%d)", c.St.Value, p.Value)
	}
	if c.Ct.sz == 1 {
		t.Fatalf("a's private field sz not preserved from merge: c.Ct.sz(%d) == b.Ct.sz(%d)", c.Ct.sz, m["sz"])
	}
	if c.Ct.Id != m["id"] {
		t.Fatalf("a's field Id not merged properly: c.Ct.Id(%s) != b.Ct.Id(%s)", c.Ct.Id, m["id"])
	}
}

func TestSimpleMap(t *testing.T) {
	a := simpleTest{}
	b := map[string]interface{}{
		"value": 42,
	}
	if err := Map(&a, b); err != nil {
		t.FailNow()
	}
	if a.Value != 42 {
		t.Fatalf("b not merged in a properly: a.Value(%d) != b.Value(%v)", a.Value, b["value"])
	}
}

type pointerMapTest struct {
	A      int
	hidden int
	B      *simpleTest
}

func TestBackAndForth(t *testing.T) {
	pt := pointerMapTest{42, 1, &simpleTest{66}}
	m := make(map[string]interface{})
	if err := Map(&m, pt); err != nil {
		t.FailNow()
	}
	var (
		v  interface{}
		ok bool
	)
	if v, ok = m["a"]; v.(int) != pt.A || !ok {
		t.Fatalf("pt not merged properly: m[`a`](%d) != pt.A(%d)", v, pt.A)
	}
	if v, ok = m["b"]; !ok {
		t.Fatalf("pt not merged properly: B is missing in m")
	}
	var st *simpleTest
	if st = v.(*simpleTest); st.Value != 66 {
		t.Fatalf("something went wrong while mapping pt on m, B wasn't copied")
	}
	bpt := pointerMapTest{}
	if err := Map(&bpt, m); err != nil {
		t.Fatal(err)
	}
	if bpt.A != pt.A {
		t.Fatalf("pt not merged properly: bpt.A(%d) != pt.A(%d)", bpt.A, pt.A)
	}
	if bpt.hidden == pt.hidden {
		t.Fatalf("pt unexpectedly merged: bpt.hidden(%d) == pt.hidden(%d)", bpt.hidden, pt.hidden)
	}
	if bpt.B.Value != pt.B.Value {
		t.Fatalf("pt not merged properly: bpt.B.Value(%d) != pt.B.Value(%d)", bpt.B.Value, pt.B.Value)
	}
}

func loadYAML(path string) (m map[string]interface{}) {
	m = make(map[string]interface{})
	raw, _ := ioutil.ReadFile(path)
	_ = yaml.Unmarshal(raw, &m)
	return
}
