package yaml_test

import (
	"errors"
	. "gopkg.in/check.v1"
	"gopkg.in/yaml.v2"
	"math"
	"net"
	"reflect"
	"strings"
	"time"
)

var unmarshalIntTest = 123

var unmarshalTests = []struct {
	data  string
	value interface{}
}{
	{
		"",
		&struct{}{},
	}, {
		"{}", &struct{}{},
	}, {
		"v: hi",
		map[string]string{"v": "hi"},
	}, {
		"v: hi", map[string]interface{}{"v": "hi"},
	}, {
		"v: true",
		map[string]string{"v": "true"},
	}, {
		"v: true",
		map[string]interface{}{"v": true},
	}, {
		"v: 10",
		map[string]interface{}{"v": 10},
	}, {
		"v: 0b10",
		map[string]interface{}{"v": 2},
	}, {
		"v: 0xA",
		map[string]interface{}{"v": 10},
	}, {
		"v: 4294967296",
		map[string]int64{"v": 4294967296},
	}, {
		"v: 0.1",
		map[string]interface{}{"v": 0.1},
	}, {
		"v: .1",
		map[string]interface{}{"v": 0.1},
	}, {
		"v: .Inf",
		map[string]interface{}{"v": math.Inf(+1)},
	}, {
		"v: -.Inf",
		map[string]interface{}{"v": math.Inf(-1)},
	}, {
		"v: -10",
		map[string]interface{}{"v": -10},
	}, {
		"v: -.1",
		map[string]interface{}{"v": -0.1},
	},

	// Simple values.
	{
		"123",
		&unmarshalIntTest,
	},

	// Floats from spec
	{
		"canonical: 6.8523e+5",
		map[string]interface{}{"canonical": 6.8523e+5},
	}, {
		"expo: 685.230_15e+03",
		map[string]interface{}{"expo": 685.23015e+03},
	}, {
		"fixed: 685_230.15",
		map[string]interface{}{"fixed": 685230.15},
	}, {
		"neginf: -.inf",
		map[string]interface{}{"neginf": math.Inf(-1)},
	}, {
		"fixed: 685_230.15",
		map[string]float64{"fixed": 685230.15},
	},
	//{"sexa: 190:20:30.15", map[string]interface{}{"sexa": 0}}, // Unsupported
	//{"notanum: .NaN", map[string]interface{}{"notanum": math.NaN()}}, // Equality of NaN fails.

	// Bools from spec
	{
		"canonical: y",
		map[string]interface{}{"canonical": true},
	}, {
		"answer: NO",
		map[string]interface{}{"answer": false},
	}, {
		"logical: True",
		map[string]interface{}{"logical": true},
	}, {
		"option: on",
		map[string]interface{}{"option": true},
	}, {
		"option: on",
		map[string]bool{"option": true},
	},
	// Ints from spec
	{
		"canonical: 685230",
		map[string]interface{}{"canonical": 685230},
	}, {
		"decimal: +685_230",
		map[string]interface{}{"decimal": 685230},
	}, {
		"octal: 02472256",
		map[string]interface{}{"octal": 685230},
	}, {
		"hexa: 0x_0A_74_AE",
		map[string]interface{}{"hexa": 685230},
	}, {
		"bin: 0b1010_0111_0100_1010_1110",
		map[string]interface{}{"bin": 685230},
	}, {
		"bin: -0b101010",
		map[string]interface{}{"bin": -42},
	}, {
		"decimal: +685_230",
		map[string]int{"decimal": 685230},
	},

	//{"sexa: 190:20:30", map[string]interface{}{"sexa": 0}}, // Unsupported

	// Nulls from spec
	{
		"empty:",
		map[string]interface{}{"empty": nil},
	}, {
		"canonical: ~",
		map[string]interface{}{"canonical": nil},
	}, {
		"english: null",
		map[string]interface{}{"english": nil},
	}, {
		"~: null key",
		map[interface{}]string{nil: "null key"},
	}, {
		"empty:",
		map[string]*bool{"empty": nil},
	},

	// Flow sequence
	{
		"seq: [A,B]",
		map[string]interface{}{"seq": []interface{}{"A", "B"}},
	}, {
		"seq: [A,B,C,]",
		map[string][]string{"seq": []string{"A", "B", "C"}},
	}, {
		"seq: [A,1,C]",
		map[string][]string{"seq": []string{"A", "1", "C"}},
	}, {
		"seq: [A,1,C]",
		map[string][]int{"seq": []int{1}},
	}, {
		"seq: [A,1,C]",
		map[string]interface{}{"seq": []interface{}{"A", 1, "C"}},
	},
	// Block sequence
	{
		"seq:\n - A\n - B",
		map[string]interface{}{"seq": []interface{}{"A", "B"}},
	}, {
		"seq:\n - A\n - B\n - C",
		map[string][]string{"seq": []string{"A", "B", "C"}},
	}, {
		"seq:\n - A\n - 1\n - C",
		map[string][]string{"seq": []string{"A", "1", "C"}},
	}, {
		"seq:\n - A\n - 1\n - C",
		map[string][]int{"seq": []int{1}},
	}, {
		"seq:\n - A\n - 1\n - C",
		map[string]interface{}{"seq": []interface{}{"A", 1, "C"}},
	},

	// Literal block scalar
	{
		"scalar: | # Comment\n\n literal\n\n \ttext\n\n",
		map[string]string{"scalar": "\nliteral\n\n\ttext\n"},
	},

	// Folded block scalar
	{
		"scalar: > # Comment\n\n folded\n line\n \n next\n line\n  * one\n  * two\n\n last\n line\n\n",
		map[string]string{"scalar": "\nfolded line\nnext line\n * one\n * two\n\nlast line\n"},
	},

	// Map inside interface with no type hints.
	{
		"a: {b: c}",
		map[interface{}]interface{}{"a": map[interface{}]interface{}{"b": "c"}},
	},

	// Structs and type conversions.
	{
		"hello: world",
		&struct{ Hello string }{"world"},
	}, {
		"a: {b: c}",
		&struct{ A struct{ B string } }{struct{ B string }{"c"}},
	}, {
		"a: {b: c}",
		&struct{ A *struct{ B string } }{&struct{ B string }{"c"}},
	}, {
		"a: {b: c}",
		&struct{ A map[string]string }{map[string]string{"b": "c"}},
	}, {
		"a: {b: c}",
		&struct{ A *map[string]string }{&map[string]string{"b": "c"}},
	}, {
		"a:",
		&struct{ A map[string]string }{},
	}, {
		"a: 1",
		&struct{ A int }{1},
	}, {
		"a: 1",
		&struct{ A float64 }{1},
	}, {
		"a: 1.0",
		&struct{ A int }{1},
	}, {
		"a: 1.0",
		&struct{ A uint }{1},
	}, {
		"a: [1, 2]",
		&struct{ A []int }{[]int{1, 2}},
	}, {
		"a: 1",
		&struct{ B int }{0},
	}, {
		"a: 1",
		&struct {
			B int "a"
		}{1},
	}, {
		"a: y",
		&struct{ A bool }{true},
	},

	// Some cross type conversions
	{
		"v: 42",
		map[string]uint{"v": 42},
	}, {
		"v: -42",
		map[string]uint{},
	}, {
		"v: 4294967296",
		map[string]uint64{"v": 4294967296},
	}, {
		"v: -4294967296",
		map[string]uint64{},
	},

	// int
	{
		"int_max: 2147483647",
		map[string]int{"int_max": math.MaxInt32},
	},
	{
		"int_min: -2147483648",
		map[string]int{"int_min": math.MinInt32},
	},
	{
		"int_overflow: 9223372036854775808", // math.MaxInt64 + 1
		map[string]int{},
	},

	// int64
	{
		"int64_max: 9223372036854775807",
		map[string]int64{"int64_max": math.MaxInt64},
	},
	{
		"int64_max_base2: 0b111111111111111111111111111111111111111111111111111111111111111",
		map[string]int64{"int64_max_base2": math.MaxInt64},
	},
	{
		"int64_min: -9223372036854775808",
		map[string]int64{"int64_min": math.MinInt64},
	},
	{
		"int64_neg_base2: -0b111111111111111111111111111111111111111111111111111111111111111",
		map[string]int64{"int64_neg_base2": -math.MaxInt64},
	},
	{
		"int64_overflow: 9223372036854775808", // math.MaxInt64 + 1
		map[string]int64{},
	},

	// uint
	{
		"uint_min: 0",
		map[string]uint{"uint_min": 0},
	},
	{
		"uint_max: 4294967295",
		map[string]uint{"uint_max": math.MaxUint32},
	},
	{
		"uint_underflow: -1",
		map[string]uint{},
	},

	// uint64
	{
		"uint64_min: 0",
		map[string]uint{"uint64_min": 0},
	},
	{
		"uint64_max: 18446744073709551615",
		map[string]uint64{"uint64_max": math.MaxUint64},
	},
	{
		"uint64_max_base2: 0b1111111111111111111111111111111111111111111111111111111111111111",
		map[string]uint64{"uint64_max_base2": math.MaxUint64},
	},
	{
		"uint64_maxint64: 9223372036854775807",
		map[string]uint64{"uint64_maxint64": math.MaxInt64},
	},
	{
		"uint64_underflow: -1",
		map[string]uint64{},
	},

	// float32
	{
		"float32_max: 3.40282346638528859811704183484516925440e+38",
		map[string]float32{"float32_max": math.MaxFloat32},
	},
	{
		"float32_nonzero: 1.401298464324817070923729583289916131280e-45",
		map[string]float32{"float32_nonzero": math.SmallestNonzeroFloat32},
	},
	{
		"float32_maxuint64: 18446744073709551615",
		map[string]float32{"float32_maxuint64": float32(math.MaxUint64)},
	},
	{
		"float32_maxuint64+1: 18446744073709551616",
		map[string]float32{"float32_maxuint64+1": float32(math.MaxUint64 + 1)},
	},

	// float64
	{
		"float64_max: 1.797693134862315708145274237317043567981e+308",
		map[string]float64{"float64_max": math.MaxFloat64},
	},
	{
		"float64_nonzero: 4.940656458412465441765687928682213723651e-324",
		map[string]float64{"float64_nonzero": math.SmallestNonzeroFloat64},
	},
	{
		"float64_maxuint64: 18446744073709551615",
		map[string]float64{"float64_maxuint64": float64(math.MaxUint64)},
	},
	{
		"float64_maxuint64+1: 18446744073709551616",
		map[string]float64{"float64_maxuint64+1": float64(math.MaxUint64 + 1)},
	},

	// Overflow cases.
	{
		"v: 4294967297",
		map[string]int32{},
	}, {
		"v: 128",
		map[string]int8{},
	},

	// Quoted values.
	{
		"'1': '\"2\"'",
		map[interface{}]interface{}{"1": "\"2\""},
	}, {
		"v:\n- A\n- 'B\n\n  C'\n",
		map[string][]string{"v": []string{"A", "B\nC"}},
	},

	// Explicit tags.
	{
		"v: !!float '1.1'",
		map[string]interface{}{"v": 1.1},
	}, {
		"v: !!null ''",
		map[string]interface{}{"v": nil},
	}, {
		"%TAG !y! tag:yaml.org,2002:\n---\nv: !y!int '1'",
		map[string]interface{}{"v": 1},
	},

	// Anchors and aliases.
	{
		"a: &x 1\nb: &y 2\nc: *x\nd: *y\n",
		&struct{ A, B, C, D int }{1, 2, 1, 2},
	}, {
		"a: &a {c: 1}\nb: *a",
		&struct {
			A, B struct {
				C int
			}
		}{struct{ C int }{1}, struct{ C int }{1}},
	}, {
		"a: &a [1, 2]\nb: *a",
		&struct{ B []int }{[]int{1, 2}},
	}, {
		"b: *a\na: &a {c: 1}",
		&struct {
			A, B struct {
				C int
			}
		}{struct{ C int }{1}, struct{ C int }{1}},
	},

	// Bug #1133337
	{
		"foo: ''",
		map[string]*string{"foo": new(string)},
	}, {
		"foo: null",
		map[string]string{"foo": ""},
	}, {
		"foo: null",
		map[string]interface{}{"foo": nil},
	},

	// Ignored field
	{
		"a: 1\nb: 2\n",
		&struct {
			A int
			B int "-"
		}{1, 0},
	},

	// Bug #1191981
	{
		"" +
			"%YAML 1.1\n" +
			"--- !!str\n" +
			`"Generic line break (no glyph)\n\` + "\n" +
			` Generic line break (glyphed)\n\` + "\n" +
			` Line separator\u2028\` + "\n" +
			` Paragraph separator\u2029"` + "\n",
		"" +
			"Generic line break (no glyph)\n" +
			"Generic line break (glyphed)\n" +
			"Line separator\u2028Paragraph separator\u2029",
	},

	// Struct inlining
	{
		"a: 1\nb: 2\nc: 3\n",
		&struct {
			A int
			C inlineB `yaml:",inline"`
		}{1, inlineB{2, inlineC{3}}},
	},

	// Map inlining
	{
		"a: 1\nb: 2\nc: 3\n",
		&struct {
			A int
			C map[string]int `yaml:",inline"`
		}{1, map[string]int{"b": 2, "c": 3}},
	},

	// bug 1243827
	{
		"a: -b_c",
		map[string]interface{}{"a": "-b_c"},
	},
	{
		"a: +b_c",
		map[string]interface{}{"a": "+b_c"},
	},
	{
		"a: 50cent_of_dollar",
		map[string]interface{}{"a": "50cent_of_dollar"},
	},

	// Duration
	{
		"a: 3s",
		map[string]time.Duration{"a": 3 * time.Second},
	},

	// Issue #24.
	{
		"a: <foo>",
		map[string]string{"a": "<foo>"},
	},

	// Base 60 floats are obsolete and unsupported.
	{
		"a: 1:1\n",
		map[string]string{"a": "1:1"},
	},

	// Binary data.
	{
		"a: !!binary gIGC\n",
		map[string]string{"a": "\x80\x81\x82"},
	}, {
		"a: !!binary |\n  " + strings.Repeat("kJCQ", 17) + "kJ\n  CQ\n",
		map[string]string{"a": strings.Repeat("\x90", 54)},
	}, {
		"a: !!binary |\n  " + strings.Repeat("A", 70) + "\n  ==\n",
		map[string]string{"a": strings.Repeat("\x00", 52)},
	},

	// Ordered maps.
	{
		"{b: 2, a: 1, d: 4, c: 3, sub: {e: 5}}",
		&yaml.MapSlice{{"b", 2}, {"a", 1}, {"d", 4}, {"c", 3}, {"sub", yaml.MapSlice{{"e", 5}}}},
	},

	// Issue #39.
	{
		"a:\n b:\n  c: d\n",
		map[string]struct{ B interface{} }{"a": {map[interface{}]interface{}{"c": "d"}}},
	},

	// Custom map type.
	{
		"a: {b: c}",
		M{"a": M{"b": "c"}},
	},

	// Support encoding.TextUnmarshaler.
	{
		"a: 1.2.3.4\n",
		map[string]net.IP{"a": net.IPv4(1, 2, 3, 4)},
	},
	{
		"a: 2015-02-24T18:19:39Z\n",
		map[string]time.Time{"a": time.Unix(1424801979, 0)},
	},

	// Encode empty lists as zero-length slices.
	{
		"a: []",
		&struct{ A []int }{[]int{}},
	},
}

type M map[interface{}]interface{}

type inlineB struct {
	B       int
	inlineC `yaml:",inline"`
}

type inlineC struct {
	C int
}

func (s *S) TestUnmarshal(c *C) {
	for _, item := range unmarshalTests {
		t := reflect.ValueOf(item.value).Type()
		var value interface{}
		switch t.Kind() {
		case reflect.Map:
			value = reflect.MakeMap(t).Interface()
		case reflect.String:
			value = reflect.New(t).Interface()
		case reflect.Ptr:
			value = reflect.New(t.Elem()).Interface()
		default:
			c.Fatalf("missing case for %s", t)
		}
		err := yaml.Unmarshal([]byte(item.data), value)
		if _, ok := err.(*yaml.TypeError); !ok {
			c.Assert(err, IsNil)
		}
		if t.Kind() == reflect.String {
			c.Assert(*value.(*string), Equals, item.value)
		} else {
			c.Assert(value, DeepEquals, item.value)
		}
	}
}

func (s *S) TestUnmarshalNaN(c *C) {
	value := map[string]interface{}{}
	err := yaml.Unmarshal([]byte("notanum: .NaN"), &value)
	c.Assert(err, IsNil)
	c.Assert(math.IsNaN(value["notanum"].(float64)), Equals, true)
}

var unmarshalErrorTests = []struct {
	data, error string
}{
	{"v: !!float 'error'", "yaml: cannot decode !!str `error` as a !!float"},
	{"v: [A,", "yaml: line 1: did not find expected node content"},
	{"v:\n- [A,", "yaml: line 2: did not find expected node content"},
	{"a: *b\n", "yaml: unknown anchor 'b' referenced"},
	{"a: &a\n  b: *a\n", "yaml: anchor 'a' value contains itself"},
	{"value: -", "yaml: block sequence entries are not allowed in this context"},
	{"a: !!binary ==", "yaml: !!binary value contains invalid base64 data"},
	{"{[.]}", `yaml: invalid map key: \[\]interface \{\}\{"\."\}`},
	{"{{.}}", `yaml: invalid map key: map\[interface\ \{\}\]interface \{\}\{".":interface \{\}\(nil\)\}`},
}

func (s *S) TestUnmarshalErrors(c *C) {
	for _, item := range unmarshalErrorTests {
		var value interface{}
		err := yaml.Unmarshal([]byte(item.data), &value)
		c.Assert(err, ErrorMatches, item.error, Commentf("Partial unmarshal: %#v", value))
	}
}

var unmarshalerTests = []struct {
	data, tag string
	value     interface{}
}{
	{"_: {hi: there}", "!!map", map[interface{}]interface{}{"hi": "there"}},
	{"_: [1,A]", "!!seq", []interface{}{1, "A"}},
	{"_: 10", "!!int", 10},
	{"_: null", "!!null", nil},
	{`_: BAR!`, "!!str", "BAR!"},
	{`_: "BAR!"`, "!!str", "BAR!"},
	{"_: !!foo 'BAR!'", "!!foo", "BAR!"},
}

var unmarshalerResult = map[int]error{}

type unmarshalerType struct {
	value interface{}
}

func (o *unmarshalerType) UnmarshalYAML(unmarshal func(v interface{}) error) error {
	if err := unmarshal(&o.value); err != nil {
		return err
	}
	if i, ok := o.value.(int); ok {
		if result, ok := unmarshalerResult[i]; ok {
			return result
		}
	}
	return nil
}

type unmarshalerPointer struct {
	Field *unmarshalerType "_"
}

type unmarshalerValue struct {
	Field unmarshalerType "_"
}

func (s *S) TestUnmarshalerPointerField(c *C) {
	for _, item := range unmarshalerTests {
		obj := &unmarshalerPointer{}
		err := yaml.Unmarshal([]byte(item.data), obj)
		c.Assert(err, IsNil)
		if item.value == nil {
			c.Assert(obj.Field, IsNil)
		} else {
			c.Assert(obj.Field, NotNil, Commentf("Pointer not initialized (%#v)", item.value))
			c.Assert(obj.Field.value, DeepEquals, item.value)
		}
	}
}

func (s *S) TestUnmarshalerValueField(c *C) {
	for _, item := range unmarshalerTests {
		obj := &unmarshalerValue{}
		err := yaml.Unmarshal([]byte(item.data), obj)
		c.Assert(err, IsNil)
		c.Assert(obj.Field, NotNil, Commentf("Pointer not initialized (%#v)", item.value))
		c.Assert(obj.Field.value, DeepEquals, item.value)
	}
}

func (s *S) TestUnmarshalerWholeDocument(c *C) {
	obj := &unmarshalerType{}
	err := yaml.Unmarshal([]byte(unmarshalerTests[0].data), obj)
	c.Assert(err, IsNil)
	value, ok := obj.value.(map[interface{}]interface{})
	c.Assert(ok, Equals, true, Commentf("value: %#v", obj.value))
	c.Assert(value["_"], DeepEquals, unmarshalerTests[0].value)
}

func (s *S) TestUnmarshalerTypeError(c *C) {
	unmarshalerResult[2] = &yaml.TypeError{[]string{"foo"}}
	unmarshalerResult[4] = &yaml.TypeError{[]string{"bar"}}
	defer func() {
		delete(unmarshalerResult, 2)
		delete(unmarshalerResult, 4)
	}()

	type T struct {
		Before int
		After  int
		M      map[string]*unmarshalerType
	}
	var v T
	data := `{before: A, m: {abc: 1, def: 2, ghi: 3, jkl: 4}, after: B}`
	err := yaml.Unmarshal([]byte(data), &v)
	c.Assert(err, ErrorMatches, ""+
		"yaml: unmarshal errors:\n"+
		"  line 1: cannot unmarshal !!str `A` into int\n"+
		"  foo\n"+
		"  bar\n"+
		"  line 1: cannot unmarshal !!str `B` into int")
	c.Assert(v.M["abc"], NotNil)
	c.Assert(v.M["def"], IsNil)
	c.Assert(v.M["ghi"], NotNil)
	c.Assert(v.M["jkl"], IsNil)

	c.Assert(v.M["abc"].value, Equals, 1)
	c.Assert(v.M["ghi"].value, Equals, 3)
}

type proxyTypeError struct{}

func (v *proxyTypeError) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	var a int32
	var b int64
	if err := unmarshal(&s); err != nil {
		panic(err)
	}
	if s == "a" {
		if err := unmarshal(&b); err == nil {
			panic("should have failed")
		}
		return unmarshal(&a)
	}
	if err := unmarshal(&a); err == nil {
		panic("should have failed")
	}
	return unmarshal(&b)
}

func (s *S) TestUnmarshalerTypeErrorProxying(c *C) {
	type T struct {
		Before int
		After  int
		M      map[string]*proxyTypeError
	}
	var v T
	data := `{before: A, m: {abc: a, def: b}, after: B}`
	err := yaml.Unmarshal([]byte(data), &v)
	c.Assert(err, ErrorMatches, ""+
		"yaml: unmarshal errors:\n"+
		"  line 1: cannot unmarshal !!str `A` into int\n"+
		"  line 1: cannot unmarshal !!str `a` into int32\n"+
		"  line 1: cannot unmarshal !!str `b` into int64\n"+
		"  line 1: cannot unmarshal !!str `B` into int")
}

type failingUnmarshaler struct{}

var failingErr = errors.New("failingErr")

func (ft *failingUnmarshaler) UnmarshalYAML(unmarshal func(interface{}) error) error {
	return failingErr
}

func (s *S) TestUnmarshalerError(c *C) {
	err := yaml.Unmarshal([]byte("a: b"), &failingUnmarshaler{})
	c.Assert(err, Equals, failingErr)
}

type sliceUnmarshaler []int

func (su *sliceUnmarshaler) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var slice []int
	err := unmarshal(&slice)
	if err == nil {
		*su = slice
		return nil
	}

	var intVal int
	err = unmarshal(&intVal)
	if err == nil {
		*su = []int{intVal}
		return nil
	}

	return err
}

func (s *S) TestUnmarshalerRetry(c *C) {
	var su sliceUnmarshaler
	err := yaml.Unmarshal([]byte("[1, 2, 3]"), &su)
	c.Assert(err, IsNil)
	c.Assert(su, DeepEquals, sliceUnmarshaler([]int{1, 2, 3}))

	err = yaml.Unmarshal([]byte("1"), &su)
	c.Assert(err, IsNil)
	c.Assert(su, DeepEquals, sliceUnmarshaler([]int{1}))
}

// From http://yaml.org/type/merge.html
var mergeTests = `
anchors:
  list:
    - &CENTER { "x": 1, "y": 2 }
    - &LEFT   { "x": 0, "y": 2 }
    - &BIG    { "r": 10 }
    - &SMALL  { "r": 1 }

# All the following maps are equal:

plain:
  # Explicit keys
  "x": 1
  "y": 2
  "r": 10
  label: center/big

mergeOne:
  # Merge one map
  << : *CENTER
  "r": 10
  label: center/big

mergeMultiple:
  # Merge multiple maps
  << : [ *CENTER, *BIG ]
  label: center/big

override:
  # Override
  << : [ *BIG, *LEFT, *SMALL ]
  "x": 1
  label: center/big

shortTag:
  # Explicit short merge tag
  !!merge "<<" : [ *CENTER, *BIG ]
  label: center/big

longTag:
  # Explicit merge long tag
  !<tag:yaml.org,2002:merge> "<<" : [ *CENTER, *BIG ]
  label: center/big

inlineMap:
  # Inlined map 
  << : {"x": 1, "y": 2, "r": 10}
  label: center/big

inlineSequenceMap:
  # Inlined map in sequence
  << : [ *CENTER, {"r": 10} ]
  label: center/big
`

func (s *S) TestMerge(c *C) {
	var want = map[interface{}]interface{}{
		"x":     1,
		"y":     2,
		"r":     10,
		"label": "center/big",
	}

	var m map[interface{}]interface{}
	err := yaml.Unmarshal([]byte(mergeTests), &m)
	c.Assert(err, IsNil)
	for name, test := range m {
		if name == "anchors" {
			continue
		}
		c.Assert(test, DeepEquals, want, Commentf("test %q failed", name))
	}
}

func (s *S) TestMergeStruct(c *C) {
	type Data struct {
		X, Y, R int
		Label   string
	}
	want := Data{1, 2, 10, "center/big"}

	var m map[string]Data
	err := yaml.Unmarshal([]byte(mergeTests), &m)
	c.Assert(err, IsNil)
	for name, test := range m {
		if name == "anchors" {
			continue
		}
		c.Assert(test, Equals, want, Commentf("test %q failed", name))
	}
}

var unmarshalNullTests = []func() interface{}{
	func() interface{} { var v interface{}; v = "v"; return &v },
	func() interface{} { var s = "s"; return &s },
	func() interface{} { var s = "s"; sptr := &s; return &sptr },
	func() interface{} { var i = 1; return &i },
	func() interface{} { var i = 1; iptr := &i; return &iptr },
	func() interface{} { m := map[string]int{"s": 1}; return &m },
	func() interface{} { m := map[string]int{"s": 1}; return m },
}

func (s *S) TestUnmarshalNull(c *C) {
	for _, test := range unmarshalNullTests {
		item := test()
		zero := reflect.Zero(reflect.TypeOf(item).Elem()).Interface()
		err := yaml.Unmarshal([]byte("null"), item)
		c.Assert(err, IsNil)
		if reflect.TypeOf(item).Kind() == reflect.Map {
			c.Assert(reflect.ValueOf(item).Interface(), DeepEquals, reflect.MakeMap(reflect.TypeOf(item)).Interface())
		} else {
			c.Assert(reflect.ValueOf(item).Elem().Interface(), DeepEquals, zero)
		}
	}
}

func (s *S) TestUnmarshalSliceOnPreset(c *C) {
	// Issue #48.
	v := struct{ A []int }{[]int{1}}
	yaml.Unmarshal([]byte("a: [2]"), &v)
	c.Assert(v.A, DeepEquals, []int{2})
}

//var data []byte
//func init() {
//	var err error
//	data, err = ioutil.ReadFile("/tmp/file.yaml")
//	if err != nil {
//		panic(err)
//	}
//}
//
//func (s *S) BenchmarkUnmarshal(c *C) {
//	var err error
//	for i := 0; i < c.N; i++ {
//		var v map[string]interface{}
//		err = yaml.Unmarshal(data, &v)
//	}
//	if err != nil {
//		panic(err)
//	}
//}
//
//func (s *S) BenchmarkMarshal(c *C) {
//	var v map[string]interface{}
//	yaml.Unmarshal(data, &v)
//	c.ResetTimer()
//	for i := 0; i < c.N; i++ {
//		yaml.Marshal(&v)
//	}
//}
