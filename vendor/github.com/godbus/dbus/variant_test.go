package dbus

import "reflect"
import "testing"

var variantFormatTests = []struct {
	v interface{}
	s string
}{
	{int32(1), `1`},
	{"foo", `"foo"`},
	{ObjectPath("/org/foo"), `@o "/org/foo"`},
	{Signature{"i"}, `@g "i"`},
	{[]byte{}, `@ay []`},
	{[]int32{1, 2}, `[1, 2]`},
	{[]int64{1, 2}, `@ax [1, 2]`},
	{[][]int32{{3, 4}, {5, 6}}, `[[3, 4], [5, 6]]`},
	{[]Variant{MakeVariant(int32(1)), MakeVariant(1.0)}, `[<1>, <@d 1>]`},
	{map[string]int32{"one": 1, "two": 2}, `{"one": 1, "two": 2}`},
	{map[int32]ObjectPath{1: "/org/foo"}, `@a{io} {1: "/org/foo"}`},
	{map[string]Variant{}, `@a{sv} {}`},
}

func TestFormatVariant(t *testing.T) {
	for i, v := range variantFormatTests {
		if s := MakeVariant(v.v).String(); s != v.s {
			t.Errorf("test %d: got %q, wanted %q", i+1, s, v.s)
		}
	}
}

var variantParseTests = []struct {
	s string
	v interface{}
}{
	{"1", int32(1)},
	{"true", true},
	{"false", false},
	{"1.0", float64(1.0)},
	{"0x10", int32(16)},
	{"1e1", float64(10)},
	{`"foo"`, "foo"},
	{`"\a\b\f\n\r\t"`, "\x07\x08\x0c\n\r\t"},
	{`"\u00e4\U0001f603"`, "\u00e4\U0001f603"},
	{"[1]", []int32{1}},
	{"[1, 2, 3]", []int32{1, 2, 3}},
	{"@ai []", []int32{}},
	{"[1, 5.0]", []float64{1, 5.0}},
	{"[[1, 2], [3, 4.0]]", [][]float64{{1, 2}, {3, 4}}},
	{`[@o "/org/foo", "/org/bar"]`, []ObjectPath{"/org/foo", "/org/bar"}},
	{"<1>", MakeVariant(int32(1))},
	{"[<1>, <2.0>]", []Variant{MakeVariant(int32(1)), MakeVariant(2.0)}},
	{`[[], [""]]`, [][]string{{}, {""}}},
	{`@a{ss} {}`, map[string]string{}},
	{`{"foo": 1}`, map[string]int32{"foo": 1}},
	{`[{}, {"foo": "bar"}]`, []map[string]string{{}, {"foo": "bar"}}},
	{`{"a": <1>, "b": <"foo">}`,
		map[string]Variant{"a": MakeVariant(int32(1)), "b": MakeVariant("foo")}},
	{`b''`, []byte{0}},
	{`b"abc"`, []byte{'a', 'b', 'c', 0}},
	{`b"\x01\0002\a\b\f\n\r\t"`, []byte{1, 2, 0x7, 0x8, 0xc, '\n', '\r', '\t', 0}},
	{`[[0], b""]`, [][]byte{{0}, {0}}},
	{"int16 0", int16(0)},
	{"byte 0", byte(0)},
}

func TestParseVariant(t *testing.T) {
	for i, v := range variantParseTests {
		nv, err := ParseVariant(v.s, Signature{})
		if err != nil {
			t.Errorf("test %d: parsing failed: %s", i+1, err)
			continue
		}
		if !reflect.DeepEqual(nv.value, v.v) {
			t.Errorf("test %d: got %q, wanted %q", i+1, nv, v.v)
		}
	}
}
