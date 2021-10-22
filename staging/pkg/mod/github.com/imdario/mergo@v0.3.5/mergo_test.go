// Copyright 2013 Dario Castañé. All rights reserved.
// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mergo

import (
	"io/ioutil"
	"reflect"
	"testing"
	"time"

	"gopkg.in/yaml.v2"
)

type simpleTest struct {
	Value int
}

type complexTest struct {
	St simpleTest
	sz int
	ID string
}

type mapTest struct {
	M map[int]int
}

type ifcTest struct {
	I interface{}
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

func TestKb(t *testing.T) {
	type testStruct struct {
		Name     string
		KeyValue map[string]interface{}
	}

	akv := make(map[string]interface{})
	akv["Key1"] = "not value 1"
	akv["Key2"] = "value2"
	a := testStruct{}
	a.Name = "A"
	a.KeyValue = akv

	bkv := make(map[string]interface{})
	bkv["Key1"] = "value1"
	bkv["Key3"] = "value3"
	b := testStruct{}
	b.Name = "B"
	b.KeyValue = bkv

	ekv := make(map[string]interface{})
	ekv["Key1"] = "value1"
	ekv["Key2"] = "value2"
	ekv["Key3"] = "value3"
	expected := testStruct{}
	expected.Name = "B"
	expected.KeyValue = ekv

	Merge(&b, a)

	if !reflect.DeepEqual(b, expected) {
		t.Errorf("Actual: %#v did not match \nExpected: %#v", b, expected)
	}
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
		t.Fatalf("b not merged in properly: a.Value(%d) != b.Value(%d)", a.Value, b.Value)
	}
	if !reflect.DeepEqual(a, b) {
		t.FailNow()
	}
}

func TestComplexStruct(t *testing.T) {
	a := complexTest{}
	a.ID = "athing"
	b := complexTest{simpleTest{42}, 1, "bthing"}
	if err := Merge(&a, b); err != nil {
		t.FailNow()
	}
	if a.St.Value != 42 {
		t.Fatalf("b not merged in properly: a.St.Value(%d) != b.St.Value(%d)", a.St.Value, b.St.Value)
	}
	if a.sz == 1 {
		t.Fatalf("a's private field sz not preserved from merge: a.sz(%d) == b.sz(%d)", a.sz, b.sz)
	}
	if a.ID == b.ID {
		t.Fatalf("a's field ID merged unexpectedly: a.ID(%s) == b.ID(%s)", a.ID, b.ID)
	}
}

func TestComplexStructWithOverwrite(t *testing.T) {
	a := complexTest{simpleTest{1}, 1, "do-not-overwrite-with-empty-value"}
	b := complexTest{simpleTest{42}, 2, ""}

	expect := complexTest{simpleTest{42}, 1, "do-not-overwrite-with-empty-value"}
	if err := MergeWithOverwrite(&a, b); err != nil {
		t.FailNow()
	}

	if !reflect.DeepEqual(a, expect) {
		t.Fatalf("Test failed:\ngot  :\n%#v\n\nwant :\n%#v\n\n", a, expect)
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
		t.Fatalf("b not merged in properly: a.C.Value(%d) != b.C.Value(%d)", a.C.Value, b.C.Value)
	}
}

type embeddingStruct struct {
	embeddedStruct
}

type embeddedStruct struct {
	A string
}

func TestEmbeddedStruct(t *testing.T) {
	tests := []struct {
		src      embeddingStruct
		dst      embeddingStruct
		expected embeddingStruct
	}{
		{
			src: embeddingStruct{
				embeddedStruct{"foo"},
			},
			dst: embeddingStruct{
				embeddedStruct{""},
			},
			expected: embeddingStruct{
				embeddedStruct{"foo"},
			},
		},
		{
			src: embeddingStruct{
				embeddedStruct{""},
			},
			dst: embeddingStruct{
				embeddedStruct{"bar"},
			},
			expected: embeddingStruct{
				embeddedStruct{"bar"},
			},
		},
		{
			src: embeddingStruct{
				embeddedStruct{"foo"},
			},
			dst: embeddingStruct{
				embeddedStruct{"bar"},
			},
			expected: embeddingStruct{
				embeddedStruct{"bar"},
			},
		},
	}

	for _, test := range tests {
		err := Merge(&test.dst, test.src)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !reflect.DeepEqual(test.dst, test.expected) {
			t.Errorf("unexpected output\nexpected:\n%+v\nsaw:\n%+v\n", test.expected, test.dst)
		}
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

func testSlice(t *testing.T, a []int, b []int, e []int, opts ...func(*Config)) {
	t.Helper()
	bc := b

	sa := sliceTest{a}
	sb := sliceTest{b}
	if err := Merge(&sa, sb, opts...); err != nil {
		t.FailNow()
	}
	if !reflect.DeepEqual(sb.S, bc) {
		t.Fatalf("Source slice was modified %d != %d", sb.S, bc)
	}
	if !reflect.DeepEqual(sa.S, e) {
		t.Fatalf("b not merged in a proper way %d != %d", sa.S, e)
	}

	ma := map[string][]int{"S": a}
	mb := map[string][]int{"S": b}
	if err := Merge(&ma, mb, opts...); err != nil {
		t.FailNow()
	}
	if !reflect.DeepEqual(mb["S"], bc) {
		t.Fatalf("map value: Source slice was modified %d != %d", mb["S"], bc)
	}
	if !reflect.DeepEqual(ma["S"], e) {
		t.Fatalf("map value: b not merged in a proper way %d != %d", ma["S"], e)
	}

	if a == nil {
		// test case with missing dst key
		ma := map[string][]int{}
		mb := map[string][]int{"S": b}
		if err := Merge(&ma, mb); err != nil {
			t.FailNow()
		}
		if !reflect.DeepEqual(mb["S"], bc) {
			t.Fatalf("missing dst key: Source slice was modified %d != %d", mb["S"], bc)
		}
		if !reflect.DeepEqual(ma["S"], e) {
			t.Fatalf("missing dst key: b not merged in a proper way %d != %d", ma["S"], e)
		}
	}

	if b == nil {
		// test case with missing src key
		ma := map[string][]int{"S": a}
		mb := map[string][]int{}
		if err := Merge(&ma, mb); err != nil {
			t.FailNow()
		}
		if !reflect.DeepEqual(mb["S"], bc) {
			t.Fatalf("missing src key: Source slice was modified %d != %d", mb["S"], bc)
		}
		if !reflect.DeepEqual(ma["S"], e) {
			t.Fatalf("missing src key: b not merged in a proper way %d != %d", ma["S"], e)
		}
	}
}

func TestSlice(t *testing.T) {
	testSlice(t, nil, []int{1, 2, 3}, []int{1, 2, 3})
	testSlice(t, []int{}, []int{1, 2, 3}, []int{1, 2, 3})
	testSlice(t, []int{1}, []int{2, 3}, []int{1})
	testSlice(t, []int{1}, []int{}, []int{1})
	testSlice(t, []int{1}, nil, []int{1})
	testSlice(t, nil, []int{1, 2, 3}, []int{1, 2, 3}, WithAppendSlice)
	testSlice(t, []int{}, []int{1, 2, 3}, []int{1, 2, 3}, WithAppendSlice)
	testSlice(t, []int{1}, []int{2, 3}, []int{1, 2, 3}, WithAppendSlice)
	testSlice(t, []int{1}, []int{}, []int{1}, WithAppendSlice)
	testSlice(t, []int{1}, nil, []int{1}, WithAppendSlice)
}

func TestEmptyMaps(t *testing.T) {
	a := mapTest{}
	b := mapTest{
		map[int]int{},
	}
	if err := Merge(&a, b); err != nil {
		t.Fail()
	}
	if !reflect.DeepEqual(a, b) {
		t.FailNow()
	}
}

func TestEmptyToEmptyMaps(t *testing.T) {
	a := mapTest{}
	b := mapTest{}
	if err := Merge(&a, b); err != nil {
		t.Fail()
	}
	if !reflect.DeepEqual(a, b) {
		t.FailNow()
	}
}

func TestEmptyToNotEmptyMaps(t *testing.T) {
	a := mapTest{map[int]int{
		1: 2,
		3: 4,
	}}
	aa := mapTest{map[int]int{
		1: 2,
		3: 4,
	}}
	b := mapTest{
		map[int]int{},
	}
	if err := Merge(&a, b); err != nil {
		t.Fail()
	}
	if !reflect.DeepEqual(a, aa) {
		t.FailNow()
	}
}

func TestMapsWithOverwrite(t *testing.T) {
	m := map[string]simpleTest{
		"a": {},   // overwritten by 16
		"b": {42}, // not overwritten by empty value
		"c": {13}, // overwritten by 12
		"d": {61},
	}
	n := map[string]simpleTest{
		"a": {16},
		"b": {},
		"c": {12},
		"e": {14},
	}
	expect := map[string]simpleTest{
		"a": {16},
		"b": {},
		"c": {12},
		"d": {61},
		"e": {14},
	}

	if err := MergeWithOverwrite(&m, n); err != nil {
		t.Fatalf(err.Error())
	}

	if !reflect.DeepEqual(m, expect) {
		t.Fatalf("Test failed:\ngot  :\n%#v\n\nwant :\n%#v\n\n", m, expect)
	}
}

func TestMaps(t *testing.T) {
	m := map[string]simpleTest{
		"a": {},
		"b": {42},
		"c": {13},
		"d": {61},
	}
	n := map[string]simpleTest{
		"a": {16},
		"b": {},
		"c": {12},
		"e": {14},
	}
	expect := map[string]simpleTest{
		"a": {0},
		"b": {42},
		"c": {13},
		"d": {61},
		"e": {14},
	}

	if err := Merge(&m, n); err != nil {
		t.Fatalf(err.Error())
	}

	if !reflect.DeepEqual(m, expect) {
		t.Fatalf("Test failed:\ngot  :\n%#v\n\nwant :\n%#v\n\n", m, expect)
	}
	if m["a"].Value != 0 {
		t.Fatalf(`n merged in m because I solved non-addressable map values TODO: m["a"].Value(%d) != n["a"].Value(%d)`, m["a"].Value, n["a"].Value)
	}
	if m["b"].Value != 42 {
		t.Fatalf(`n wrongly merged in m: m["b"].Value(%d) != n["b"].Value(%d)`, m["b"].Value, n["b"].Value)
	}
	if m["c"].Value != 13 {
		t.Fatalf(`n overwritten in m: m["c"].Value(%d) != n["c"].Value(%d)`, m["c"].Value, n["c"].Value)
	}
}

func TestMapsWithNilPointer(t *testing.T) {
	m := map[string]*simpleTest{
		"a": nil,
		"b": nil,
	}
	n := map[string]*simpleTest{
		"b": nil,
		"c": nil,
	}
	expect := map[string]*simpleTest{
		"a": nil,
		"b": nil,
		"c": nil,
	}

	if err := Merge(&m, n, WithOverride); err != nil {
		t.Fatalf(err.Error())
	}

	if !reflect.DeepEqual(m, expect) {
		t.Fatalf("Test failed:\ngot   :\n%#v\n\nwant :\n%#v\n\n", m, expect)
	}
}

func TestYAMLMaps(t *testing.T) {
	thing := loadYAML("testdata/thing.yml")
	license := loadYAML("testdata/license.yml")
	ft := thing["fields"].(map[interface{}]interface{})
	fl := license["fields"].(map[interface{}]interface{})
	// license has one extra field (site) and another already existing in thing (author) that Mergo won't override.
	expectedLength := len(ft) + len(fl) - 1
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
	a.ID = "athing"
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
		t.Fatalf("b not merged in properly: c.Ct.St.Value(%d) != b.Ct.St.Value(%d)", c.Ct.St.Value, n["value"])
	}
	if c.St.Value != 144 {
		t.Fatalf("b not merged in properly: c.St.Value(%d) != b.St.Value(%d)", c.St.Value, o.Value)
	}
	if c.Nt.Value != 3 {
		t.Fatalf("b not merged in properly: c.Nt.Value(%d) != b.Nt.Value(%d)", c.St.Value, p.Value)
	}
	if c.Ct.sz == 1 {
		t.Fatalf("a's private field sz not preserved from merge: c.Ct.sz(%d) == b.Ct.sz(%d)", c.Ct.sz, m["sz"])
	}
	if c.Ct.ID == m["id"] {
		t.Fatalf("a's field ID merged unexpectedly: c.Ct.ID(%s) == b.Ct.ID(%s)", c.Ct.ID, m["id"])
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
		t.Fatalf("b not merged in properly: a.Value(%d) != b.Value(%v)", a.Value, b["value"])
	}
}

func TestIfcMap(t *testing.T) {
	a := ifcTest{}
	b := ifcTest{42}
	if err := Map(&a, b); err != nil {
		t.FailNow()
	}
	if a.I != 42 {
		t.Fatalf("b not merged in properly: a.I(%d) != b.I(%d)", a.I, b.I)
	}
	if !reflect.DeepEqual(a, b) {
		t.FailNow()
	}
}

func TestIfcMapNoOverwrite(t *testing.T) {
	a := ifcTest{13}
	b := ifcTest{42}
	if err := Map(&a, b); err != nil {
		t.FailNow()
	}
	if a.I != 13 {
		t.Fatalf("a not left alone: a.I(%d) == b.I(%d)", a.I, b.I)
	}
}

func TestIfcMapWithOverwrite(t *testing.T) {
	a := ifcTest{13}
	b := ifcTest{42}
	if err := MapWithOverwrite(&a, b); err != nil {
		t.FailNow()
	}
	if a.I != 42 {
		t.Fatalf("b not merged in properly: a.I(%d) != b.I(%d)", a.I, b.I)
	}
	if !reflect.DeepEqual(a, b) {
		t.FailNow()
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
		t.Fatalf("pt not merged in properly: m[`a`](%d) != pt.A(%d)", v, pt.A)
	}
	if v, ok = m["b"]; !ok {
		t.Fatalf("pt not merged in properly: B is missing in m")
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
		t.Fatalf("pt not merged in properly: bpt.A(%d) != pt.A(%d)", bpt.A, pt.A)
	}
	if bpt.hidden == pt.hidden {
		t.Fatalf("pt unexpectedly merged: bpt.hidden(%d) == pt.hidden(%d)", bpt.hidden, pt.hidden)
	}
	if bpt.B.Value != pt.B.Value {
		t.Fatalf("pt not merged in properly: bpt.B.Value(%d) != pt.B.Value(%d)", bpt.B.Value, pt.B.Value)
	}
}

func TestEmbeddedPointerUnpacking(t *testing.T) {
	tests := []struct{ input pointerMapTest }{
		{pointerMapTest{42, 1, nil}},
		{pointerMapTest{42, 1, &simpleTest{66}}},
	}
	newValue := 77
	m := map[string]interface{}{
		"b": map[string]interface{}{
			"value": newValue,
		},
	}
	for _, test := range tests {
		pt := test.input
		if err := MapWithOverwrite(&pt, m); err != nil {
			t.FailNow()
		}
		if pt.B.Value != newValue {
			t.Fatalf("pt not mapped properly: pt.A.Value(%d) != m[`b`][`value`](%d)", pt.B.Value, newValue)
		}

	}
}

type structWithTimePointer struct {
	Birth *time.Time
}

func TestTime(t *testing.T) {
	now := time.Now()
	dataStruct := structWithTimePointer{
		Birth: &now,
	}
	dataMap := map[string]interface{}{
		"Birth": &now,
	}
	b := structWithTimePointer{}
	if err := Merge(&b, dataStruct); err != nil {
		t.FailNow()
	}
	if b.Birth.IsZero() {
		t.Fatalf("time.Time not merged in properly: b.Birth(%v) != dataStruct['Birth'](%v)", b.Birth, dataStruct.Birth)
	}
	if b.Birth != dataStruct.Birth {
		t.Fatalf("time.Time not merged in properly: b.Birth(%v) != dataStruct['Birth'](%v)", b.Birth, dataStruct.Birth)
	}
	b = structWithTimePointer{}
	if err := Map(&b, dataMap); err != nil {
		t.FailNow()
	}
	if b.Birth.IsZero() {
		t.Fatalf("time.Time not merged in properly: b.Birth(%v) != dataMap['Birth'](%v)", b.Birth, dataMap["Birth"])
	}
}

type simpleNested struct {
	A int
}

type structWithNestedPtrValueMap struct {
	NestedPtrValue map[string]*simpleNested
}

func TestNestedPtrValueInMap(t *testing.T) {
	src := &structWithNestedPtrValueMap{
		NestedPtrValue: map[string]*simpleNested{
			"x": {
				A: 1,
			},
		},
	}
	dst := &structWithNestedPtrValueMap{
		NestedPtrValue: map[string]*simpleNested{
			"x": {},
		},
	}
	if err := Map(dst, src); err != nil {
		t.FailNow()
	}
	if dst.NestedPtrValue["x"].A == 0 {
		t.Fatalf("Nested Ptr value not merged in properly: dst.NestedPtrValue[\"x\"].A(%v) != src.NestedPtrValue[\"x\"].A(%v)", dst.NestedPtrValue["x"].A, src.NestedPtrValue["x"].A)
	}
}

func loadYAML(path string) (m map[string]interface{}) {
	m = make(map[string]interface{})
	raw, _ := ioutil.ReadFile(path)
	_ = yaml.Unmarshal(raw, &m)
	return
}

type structWithMap struct {
	m map[string]structWithUnexportedProperty
}

type structWithUnexportedProperty struct {
	s string
}

func TestUnexportedProperty(t *testing.T) {
	a := structWithMap{map[string]structWithUnexportedProperty{
		"key": {"hello"},
	}}
	b := structWithMap{map[string]structWithUnexportedProperty{
		"key": {"hi"},
	}}
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Should not have panicked")
		}
	}()
	Merge(&a, b)
}

type structWithBoolPointer struct {
	C *bool
}

func TestBooleanPointer(t *testing.T) {
	bt, bf := true, false
	src := structWithBoolPointer{
		&bt,
	}
	dst := structWithBoolPointer{
		&bf,
	}
	if err := Merge(&dst, src); err != nil {
		t.FailNow()
	}
	if dst.C == src.C {
		t.Fatalf("dst.C should be a different pointer than src.C")
	}
	if *dst.C != *src.C {
		t.Fatalf("dst.C should be true")
	}
}
