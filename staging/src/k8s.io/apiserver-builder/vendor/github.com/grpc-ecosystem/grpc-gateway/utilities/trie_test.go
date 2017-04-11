package utilities_test

import (
	"reflect"
	"testing"

	"github.com/grpc-ecosystem/grpc-gateway/utilities"
)

func TestMaxCommonPrefix(t *testing.T) {
	for _, spec := range []struct {
		da     utilities.DoubleArray
		tokens []string
		want   bool
	}{
		{
			da:     utilities.DoubleArray{},
			tokens: nil,
			want:   false,
		},
		{
			da:     utilities.DoubleArray{},
			tokens: []string{"foo"},
			want:   false,
		},
		{
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
				},
				Base:  []int{1, 1, 0},
				Check: []int{0, 1, 2},
			},
			tokens: nil,
			want:   false,
		},
		{
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
				},
				Base:  []int{1, 1, 0},
				Check: []int{0, 1, 2},
			},
			tokens: []string{"foo"},
			want:   true,
		},
		{
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
				},
				Base:  []int{1, 1, 0},
				Check: []int{0, 1, 2},
			},
			tokens: []string{"bar"},
			want:   false,
		},
		{
			// foo|bar
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
				},
				Base:  []int{1, 1, 2, 0, 0},
				Check: []int{0, 1, 1, 2, 3},
				// 0: ^
				// 1: ^foo
				// 2: ^bar
				// 3: ^foo$
				// 4: ^bar$
			},
			tokens: []string{"foo"},
			want:   true,
		},
		{
			// foo|bar
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
				},
				Base:  []int{1, 1, 2, 0, 0},
				Check: []int{0, 1, 1, 2, 3},
				// 0: ^
				// 1: ^foo
				// 2: ^bar
				// 3: ^foo$
				// 4: ^bar$
			},
			tokens: []string{"bar"},
			want:   true,
		},
		{
			// foo|bar
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
				},
				Base:  []int{1, 1, 2, 0, 0},
				Check: []int{0, 1, 1, 2, 3},
				// 0: ^
				// 1: ^foo
				// 2: ^bar
				// 3: ^foo$
				// 4: ^bar$
			},
			tokens: []string{"something-else"},
			want:   false,
		},
		{
			// foo|bar
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
				},
				Base:  []int{1, 1, 2, 0, 0},
				Check: []int{0, 1, 1, 2, 3},
				// 0: ^
				// 1: ^foo
				// 2: ^bar
				// 3: ^foo$
				// 4: ^bar$
			},
			tokens: []string{"foo", "bar"},
			want:   true,
		},
		{
			// foo|foo\.bar|bar
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
				},
				Base:  []int{1, 3, 1, 0, 4, 0, 0},
				Check: []int{0, 1, 1, 3, 2, 2, 5},
				// 0: ^
				// 1: ^foo
				// 2: ^bar
				// 3: ^bar$
				// 4: ^foo.bar
				// 5: ^foo$
				// 6: ^foo.bar$
			},
			tokens: []string{"foo"},
			want:   true,
		},
		{
			// foo|foo\.bar|bar
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
				},
				Base:  []int{1, 3, 1, 0, 4, 0, 0},
				Check: []int{0, 1, 1, 3, 2, 2, 5},
				// 0: ^
				// 1: ^foo
				// 2: ^bar
				// 3: ^bar$
				// 4: ^foo.bar
				// 5: ^foo$
				// 6: ^foo.bar$
			},
			tokens: []string{"foo", "bar"},
			want:   true,
		},
		{
			// foo|foo\.bar|bar
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
				},
				Base:  []int{1, 3, 1, 0, 4, 0, 0},
				Check: []int{0, 1, 1, 3, 2, 2, 5},
				// 0: ^
				// 1: ^foo
				// 2: ^bar
				// 3: ^bar$
				// 4: ^foo.bar
				// 5: ^foo$
				// 6: ^foo.bar$
			},
			tokens: []string{"bar"},
			want:   true,
		},
		{
			// foo|foo\.bar|bar
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
				},
				Base:  []int{1, 3, 1, 0, 4, 0, 0},
				Check: []int{0, 1, 1, 3, 2, 2, 5},
				// 0: ^
				// 1: ^foo
				// 2: ^bar
				// 3: ^bar$
				// 4: ^foo.bar
				// 5: ^foo$
				// 6: ^foo.bar$
			},
			tokens: []string{"something-else"},
			want:   false,
		},
		{
			// foo|foo\.bar|bar
			da: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
				},
				Base:  []int{1, 3, 1, 0, 4, 0, 0},
				Check: []int{0, 1, 1, 3, 2, 2, 5},
				// 0: ^
				// 1: ^foo
				// 2: ^bar
				// 3: ^bar$
				// 4: ^foo.bar
				// 5: ^foo$
				// 6: ^foo.bar$
			},
			tokens: []string{"foo", "bar", "baz"},
			want:   true,
		},
	} {
		got := spec.da.HasCommonPrefix(spec.tokens)
		if got != spec.want {
			t.Errorf("%#v.HasCommonPrefix(%v) = %v; want %v", spec.da, spec.tokens, got, spec.want)
		}
	}
}

func TestAdd(t *testing.T) {
	for _, spec := range []struct {
		tokens [][]string
		want   utilities.DoubleArray
	}{
		{
			want: utilities.DoubleArray{
				Encoding: make(map[string]int),
			},
		},
		{
			tokens: [][]string{{"foo"}},
			want: utilities.DoubleArray{
				Encoding: map[string]int{"foo": 0},
				Base:     []int{1, 1, 0},
				Check:    []int{0, 1, 2},
				// 0: ^
				// 1: ^foo
				// 2: ^foo$
			},
		},
		{
			tokens: [][]string{{"foo"}, {"bar"}},
			want: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
				},
				Base:  []int{1, 1, 2, 0, 0},
				Check: []int{0, 1, 1, 2, 3},
				// 0: ^
				// 1: ^foo
				// 2: ^bar
				// 3: ^foo$
				// 4: ^bar$
			},
		},
		{
			tokens: [][]string{{"foo", "bar"}, {"foo", "baz"}},
			want: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
					"baz": 2,
				},
				Base:  []int{1, 1, 1, 2, 0, 0},
				Check: []int{0, 1, 2, 2, 3, 4},
				// 0: ^
				// 1: ^foo
				// 2: ^foo.bar
				// 3: ^foo.baz
				// 4: ^foo.bar$
				// 5: ^foo.baz$
			},
		},
		{
			tokens: [][]string{{"foo", "bar"}, {"foo", "baz"}, {"qux"}},
			want: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
					"baz": 2,
					"qux": 3,
				},
				Base:  []int{1, 1, 1, 2, 3, 0, 0, 0},
				Check: []int{0, 1, 2, 2, 1, 3, 4, 5},
				// 0: ^
				// 1: ^foo
				// 2: ^foo.bar
				// 3: ^foo.baz
				// 4: ^qux
				// 5: ^foo.bar$
				// 6: ^foo.baz$
				// 7: ^qux$
			},
		},
		{
			tokens: [][]string{
				{"foo", "bar"},
				{"foo", "baz", "bar"},
				{"qux", "foo"},
			},
			want: utilities.DoubleArray{
				Encoding: map[string]int{
					"foo": 0,
					"bar": 1,
					"baz": 2,
					"qux": 3,
				},
				Base:  []int{1, 1, 1, 5, 8, 0, 3, 0, 5, 0},
				Check: []int{0, 1, 2, 2, 1, 3, 4, 7, 5, 9},
				// 0: ^
				// 1: ^foo
				// 2: ^foo.bar
				// 3: ^foo.baz
				// 4: ^qux
				// 5: ^foo.bar$
				// 6: ^foo.baz.bar
				// 7: ^foo.baz.bar$
				// 8: ^qux.foo
				// 9: ^qux.foo$
			},
		},
	} {
		da := utilities.NewDoubleArray(spec.tokens)
		if got, want := da.Encoding, spec.want.Encoding; !reflect.DeepEqual(got, want) {
			t.Errorf("da.Encoding = %v; want %v; tokens = %#v", got, want, spec.tokens)
		}
		if got, want := da.Base, spec.want.Base; !compareArray(got, want) {
			t.Errorf("da.Base = %v; want %v; tokens = %#v", got, want, spec.tokens)
		}
		if got, want := da.Check, spec.want.Check; !compareArray(got, want) {
			t.Errorf("da.Check = %v; want %v; tokens = %#v", got, want, spec.tokens)
		}
	}
}

func compareArray(got, want []int) bool {
	var i int
	for i = 0; i < len(got) && i < len(want); i++ {
		if got[i] != want[i] {
			return false
		}
	}
	if i < len(want) {
		return false
	}
	for ; i < len(got); i++ {
		if got[i] != 0 {
			return false
		}
	}
	return true
}
