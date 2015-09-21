package pretty

import (
	"fmt"
	"io"
	"testing"
	"unsafe"
)

type test struct {
	v interface{}
	s string
}

type LongStructTypeName struct {
	longFieldName      interface{}
	otherLongFieldName interface{}
}

type SA struct {
	t *T
}

type T struct {
	x, y int
}

type F int

func (f F) Format(s fmt.State, c rune) {
	fmt.Fprintf(s, "F(%d)", int(f))
}

var long = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

var gosyntax = []test{
	{nil, `nil`},
	{"", `""`},
	{"a", `"a"`},
	{1, "int(1)"},
	{1.0, "float64(1)"},
	{[]int(nil), "[]int(nil)"},
	{[0]int{}, "[0]int{}"},
	{complex(1, 0), "(1+0i)"},
	//{make(chan int), "(chan int)(0x1234)"},
	{unsafe.Pointer(uintptr(1)), "unsafe.Pointer(0x1)"},
	{func(int) {}, "func(int) {...}"},
	{map[int]int{1: 1}, "map[int]int{1:1}"},
	{int32(1), "int32(1)"},
	{io.EOF, `&errors.errorString{s:"EOF"}`},
	{[]string{"a"}, `[]string{"a"}`},
	{
		[]string{long},
		`[]string{"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}`,
	},
	{F(5), "pretty.F(5)"},
	{
		SA{&T{1, 2}},
		`pretty.SA{
    t:  &pretty.T{x:1, y:2},
}`,
	},
	{
		map[int][]byte{1: []byte{}},
		`map[int][]uint8{
    1:  {},
}`,
	},
	{
		map[int]T{1: T{}},
		`map[int]pretty.T{
    1:  {},
}`,
	},
	{
		long,
		`"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"`,
	},
	{
		LongStructTypeName{
			longFieldName:      LongStructTypeName{},
			otherLongFieldName: long,
		},
		`pretty.LongStructTypeName{
    longFieldName:      pretty.LongStructTypeName{},
    otherLongFieldName: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
}`,
	},
	{
		&LongStructTypeName{
			longFieldName:      &LongStructTypeName{},
			otherLongFieldName: (*LongStructTypeName)(nil),
		},
		`&pretty.LongStructTypeName{
    longFieldName:      &pretty.LongStructTypeName{},
    otherLongFieldName: (*pretty.LongStructTypeName)(nil),
}`,
	},
	{
		[]LongStructTypeName{
			{nil, nil},
			{3, 3},
			{long, nil},
		},
		`[]pretty.LongStructTypeName{
    {},
    {
        longFieldName:      int(3),
        otherLongFieldName: int(3),
    },
    {
        longFieldName:      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        otherLongFieldName: nil,
    },
}`,
	},
	{
		[]interface{}{
			LongStructTypeName{nil, nil},
			[]byte{1, 2, 3},
			T{3, 4},
			LongStructTypeName{long, nil},
		},
		`[]interface {}{
    pretty.LongStructTypeName{},
    []uint8{0x1, 0x2, 0x3},
    pretty.T{x:3, y:4},
    pretty.LongStructTypeName{
        longFieldName:      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        otherLongFieldName: nil,
    },
}`,
	},
}

func TestGoSyntax(t *testing.T) {
	for _, tt := range gosyntax {
		s := fmt.Sprintf("%# v", Formatter(tt.v))
		if tt.s != s {
			t.Errorf("expected %q", tt.s)
			t.Errorf("got      %q", s)
			t.Errorf("expraw\n%s", tt.s)
			t.Errorf("gotraw\n%s", s)
		}
	}
}
