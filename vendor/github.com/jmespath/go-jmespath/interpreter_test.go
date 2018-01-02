package jmespath

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

type scalars struct {
	Foo string
	Bar string
}

type sliceType struct {
	A string
	B []scalars
	C []*scalars
}

type benchmarkStruct struct {
	Fooasdfasdfasdfasdf string
}

type benchmarkNested struct {
	Fooasdfasdfasdfasdf nestedA
}

type nestedA struct {
	Fooasdfasdfasdfasdf nestedB
}

type nestedB struct {
	Fooasdfasdfasdfasdf nestedC
}

type nestedC struct {
	Fooasdfasdfasdfasdf string
}

type nestedSlice struct {
	A []sliceType
}

func TestCanSupportEmptyInterface(t *testing.T) {
	assert := assert.New(t)
	data := make(map[string]interface{})
	data["foo"] = "bar"
	result, err := Search("foo", data)
	assert.Nil(err)
	assert.Equal("bar", result)
}

func TestCanSupportUserDefinedStructsValue(t *testing.T) {
	assert := assert.New(t)
	s := scalars{Foo: "one", Bar: "bar"}
	result, err := Search("Foo", s)
	assert.Nil(err)
	assert.Equal("one", result)
}

func TestCanSupportUserDefinedStructsRef(t *testing.T) {
	assert := assert.New(t)
	s := scalars{Foo: "one", Bar: "bar"}
	result, err := Search("Foo", &s)
	assert.Nil(err)
	assert.Equal("one", result)
}

func TestCanSupportStructWithSliceAll(t *testing.T) {
	assert := assert.New(t)
	data := sliceType{A: "foo", B: []scalars{{"f1", "b1"}, {"correct", "b2"}}}
	result, err := Search("B[].Foo", data)
	assert.Nil(err)
	assert.Equal([]interface{}{"f1", "correct"}, result)
}

func TestCanSupportStructWithSlicingExpression(t *testing.T) {
	assert := assert.New(t)
	data := sliceType{A: "foo", B: []scalars{{"f1", "b1"}, {"correct", "b2"}}}
	result, err := Search("B[:].Foo", data)
	assert.Nil(err)
	assert.Equal([]interface{}{"f1", "correct"}, result)
}

func TestCanSupportStructWithFilterProjection(t *testing.T) {
	assert := assert.New(t)
	data := sliceType{A: "foo", B: []scalars{{"f1", "b1"}, {"correct", "b2"}}}
	result, err := Search("B[? `true` ].Foo", data)
	assert.Nil(err)
	assert.Equal([]interface{}{"f1", "correct"}, result)
}

func TestCanSupportStructWithSlice(t *testing.T) {
	assert := assert.New(t)
	data := sliceType{A: "foo", B: []scalars{{"f1", "b1"}, {"correct", "b2"}}}
	result, err := Search("B[-1].Foo", data)
	assert.Nil(err)
	assert.Equal("correct", result)
}

func TestCanSupportStructWithOrExpressions(t *testing.T) {
	assert := assert.New(t)
	data := sliceType{A: "foo", C: nil}
	result, err := Search("C || A", data)
	assert.Nil(err)
	assert.Equal("foo", result)
}

func TestCanSupportStructWithSlicePointer(t *testing.T) {
	assert := assert.New(t)
	data := sliceType{A: "foo", C: []*scalars{{"f1", "b1"}, {"correct", "b2"}}}
	result, err := Search("C[-1].Foo", data)
	assert.Nil(err)
	assert.Equal("correct", result)
}

func TestWillAutomaticallyCapitalizeFieldNames(t *testing.T) {
	assert := assert.New(t)
	s := scalars{Foo: "one", Bar: "bar"}
	// Note that there's a lower cased "foo" instead of "Foo",
	// but it should still correspond to the Foo field in the
	// scalars struct
	result, err := Search("foo", &s)
	assert.Nil(err)
	assert.Equal("one", result)
}

func TestCanSupportStructWithSliceLowerCased(t *testing.T) {
	assert := assert.New(t)
	data := sliceType{A: "foo", B: []scalars{{"f1", "b1"}, {"correct", "b2"}}}
	result, err := Search("b[-1].foo", data)
	assert.Nil(err)
	assert.Equal("correct", result)
}

func TestCanSupportStructWithNestedPointers(t *testing.T) {
	assert := assert.New(t)
	data := struct{ A *struct{ B int } }{}
	result, err := Search("A.B", data)
	assert.Nil(err)
	assert.Nil(result)
}

func TestCanSupportFlattenNestedSlice(t *testing.T) {
	assert := assert.New(t)
	data := nestedSlice{A: []sliceType{
		{B: []scalars{{Foo: "f1a"}, {Foo: "f1b"}}},
		{B: []scalars{{Foo: "f2a"}, {Foo: "f2b"}}},
	}}
	result, err := Search("A[].B[].Foo", data)
	assert.Nil(err)
	assert.Equal([]interface{}{"f1a", "f1b", "f2a", "f2b"}, result)
}

func TestCanSupportFlattenNestedEmptySlice(t *testing.T) {
	assert := assert.New(t)
	data := nestedSlice{A: []sliceType{
		{}, {B: []scalars{{Foo: "a"}}},
	}}
	result, err := Search("A[].B[].Foo", data)
	assert.Nil(err)
	assert.Equal([]interface{}{"a"}, result)
}

func TestCanSupportProjectionsWithStructs(t *testing.T) {
	assert := assert.New(t)
	data := nestedSlice{A: []sliceType{
		{A: "first"}, {A: "second"}, {A: "third"},
	}}
	result, err := Search("A[*].A", data)
	assert.Nil(err)
	assert.Equal([]interface{}{"first", "second", "third"}, result)
}

func TestCanSupportSliceOfStructsWithFunctions(t *testing.T) {
	assert := assert.New(t)
	data := []scalars{scalars{"a1", "b1"}, scalars{"a2", "b2"}}
	result, err := Search("length(@)", data)
	assert.Nil(err)
	assert.Equal(result.(float64), 2.0)
}

func BenchmarkInterpretSingleFieldStruct(b *testing.B) {
	intr := newInterpreter()
	parser := NewParser()
	ast, _ := parser.Parse("fooasdfasdfasdfasdf")
	data := benchmarkStruct{"foobarbazqux"}
	for i := 0; i < b.N; i++ {
		intr.Execute(ast, &data)
	}
}

func BenchmarkInterpretNestedStruct(b *testing.B) {
	intr := newInterpreter()
	parser := NewParser()
	ast, _ := parser.Parse("fooasdfasdfasdfasdf.fooasdfasdfasdfasdf.fooasdfasdfasdfasdf.fooasdfasdfasdfasdf")
	data := benchmarkNested{
		nestedA{
			nestedB{
				nestedC{"foobarbazqux"},
			},
		},
	}
	for i := 0; i < b.N; i++ {
		intr.Execute(ast, &data)
	}
}

func BenchmarkInterpretNestedMaps(b *testing.B) {
	jsonData := []byte(`{"fooasdfasdfasdfasdf": {"fooasdfasdfasdfasdf": {"fooasdfasdfasdfasdf": {"fooasdfasdfasdfasdf": "foobarbazqux"}}}}`)
	var data interface{}
	json.Unmarshal(jsonData, &data)

	intr := newInterpreter()
	parser := NewParser()
	ast, _ := parser.Parse("fooasdfasdfasdfasdf.fooasdfasdfasdfasdf.fooasdfasdfasdfasdf.fooasdfasdfasdfasdf")
	for i := 0; i < b.N; i++ {
		intr.Execute(ast, data)
	}
}
