// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmp_test

import (
	"bytes"
	"crypto/md5"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/go-cmp/cmp/internal/flags"

	pb "github.com/google/go-cmp/cmp/internal/testprotos"
	ts "github.com/google/go-cmp/cmp/internal/teststructs"
	foo1 "github.com/google/go-cmp/cmp/internal/teststructs/foo1"
	foo2 "github.com/google/go-cmp/cmp/internal/teststructs/foo2"
)

func init() {
	flags.Deterministic = true
}

var update = flag.Bool("update", false, "update golden test files")

const goldenHeaderPrefix = "<<< "
const goldenFooterPrefix = ">>> "

/// mustParseGolden parses a file as a set of key-value pairs.
//
// The syntax is simple and looks something like:
//
//	<<< Key1
//	value1a
//	value1b
//	>>> Key1
//	<<< Key2
//	value2
//	>>> Key2
//
// It is the user's responsibility to choose a sufficiently unique key name
// such that it never appears in the body of the value itself.
func mustParseGolden(path string) map[string]string {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		panic(err)
	}
	s := string(b)

	out := map[string]string{}
	for len(s) > 0 {
		// Identify the next header.
		i := strings.Index(s, "\n") + len("\n")
		header := s[:i]
		if !strings.HasPrefix(header, goldenHeaderPrefix) {
			panic(fmt.Sprintf("invalid header: %q", header))
		}

		// Locate the next footer.
		footer := goldenFooterPrefix + header[len(goldenHeaderPrefix):]
		j := strings.Index(s, footer)
		if j < 0 {
			panic(fmt.Sprintf("missing footer: %q", footer))
		}

		// Store the name and data.
		name := header[len(goldenHeaderPrefix) : len(header)-len("\n")]
		if _, ok := out[name]; ok {
			panic(fmt.Sprintf("duplicate name: %q", name))
		}
		out[name] = s[len(header):j]
		s = s[j+len(footer):]
	}
	return out
}
func mustFormatGolden(path string, in []struct{ Name, Data string }) {
	var b []byte
	for _, v := range in {
		b = append(b, goldenHeaderPrefix+v.Name+"\n"...)
		b = append(b, v.Data...)
		b = append(b, goldenFooterPrefix+v.Name+"\n"...)
	}
	if err := ioutil.WriteFile(path, b, 0664); err != nil {
		panic(err)
	}
}

var now = time.Date(2009, time.November, 10, 23, 00, 00, 00, time.UTC)

func newInt(n int) *int { return &n }

type Stringer string

func newStringer(s string) fmt.Stringer { return (*Stringer)(&s) }
func (s Stringer) String() string       { return string(s) }

type test struct {
	label     string       // Test name
	x, y      interface{}  // Input values to compare
	opts      []cmp.Option // Input options
	wantEqual bool         // Whether any difference is expected
	wantPanic string       // Sub-string of an expected panic message
	reason    string       // The reason for the expected outcome
}

func TestDiff(t *testing.T) {
	var tests []test
	tests = append(tests, comparerTests()...)
	tests = append(tests, transformerTests()...)
	tests = append(tests, reporterTests()...)
	tests = append(tests, embeddedTests()...)
	tests = append(tests, methodTests()...)
	tests = append(tests, cycleTests()...)
	tests = append(tests, project1Tests()...)
	tests = append(tests, project2Tests()...)
	tests = append(tests, project3Tests()...)
	tests = append(tests, project4Tests()...)

	const goldenFile = "testdata/diffs"
	gotDiffs := []struct{ Name, Data string }{}
	wantDiffs := mustParseGolden(goldenFile)
	for _, tt := range tests {
		tt := tt
		t.Run(tt.label, func(t *testing.T) {
			if !*update {
				t.Parallel()
			}
			var gotDiff, gotPanic string
			func() {
				defer func() {
					if ex := recover(); ex != nil {
						if s, ok := ex.(string); ok {
							gotPanic = s
						} else {
							panic(ex)
						}
					}
				}()
				gotDiff = cmp.Diff(tt.x, tt.y, tt.opts...)
			}()

			switch {
			case strings.Contains(t.Name(), "#"):
				panic("unique test name must be provided")
			case tt.reason == "":
				panic("reason must be provided")
			case tt.wantPanic == "":
				if gotPanic != "" {
					t.Fatalf("unexpected panic message: %s\nreason: %v", gotPanic, tt.reason)
				}
				if *update {
					if gotDiff != "" {
						gotDiffs = append(gotDiffs, struct{ Name, Data string }{t.Name(), gotDiff})
					}
				} else {
					wantDiff := wantDiffs[t.Name()]
					if diff := cmp.Diff(wantDiff, gotDiff); diff != "" {
						t.Fatalf("Diff:\ngot:\n%s\nwant:\n%s\ndiff: (-want +got)\n%s\nreason: %v", gotDiff, wantDiff, diff, tt.reason)
					}
				}
				gotEqual := gotDiff == ""
				if gotEqual != tt.wantEqual {
					t.Fatalf("Equal = %v, want %v\nreason: %v", gotEqual, tt.wantEqual, tt.reason)
				}
			default:
				if !strings.Contains(gotPanic, tt.wantPanic) {
					t.Fatalf("panic message:\ngot:  %s\nwant: %s\nreason: %v", gotPanic, tt.wantPanic, tt.reason)
				}
			}
		})
	}

	if *update {
		mustFormatGolden(goldenFile, gotDiffs)
	}
}

func comparerTests() []test {
	const label = "Comparer"

	type Iface1 interface {
		Method()
	}
	type Iface2 interface {
		Method()
	}

	type tarHeader struct {
		Name       string
		Mode       int64
		Uid        int
		Gid        int
		Size       int64
		ModTime    time.Time
		Typeflag   byte
		Linkname   string
		Uname      string
		Gname      string
		Devmajor   int64
		Devminor   int64
		AccessTime time.Time
		ChangeTime time.Time
		Xattrs     map[string]string
	}

	type namedWithUnexported struct {
		unexported string
	}

	makeTarHeaders := func(tf byte) (hs []tarHeader) {
		for i := 0; i < 5; i++ {
			hs = append(hs, tarHeader{
				Name: fmt.Sprintf("some/dummy/test/file%d", i),
				Mode: 0664, Uid: i * 1000, Gid: i * 1000, Size: 1 << uint(i),
				ModTime: now.Add(time.Duration(i) * time.Hour),
				Uname:   "user", Gname: "group",
				Typeflag: tf,
			})
		}
		return hs
	}

	return []test{{
		label:     label + "/Nil",
		x:         nil,
		y:         nil,
		wantEqual: true,
		reason:    "nils are equal",
	}, {
		label:     label + "/Integer",
		x:         1,
		y:         1,
		wantEqual: true,
		reason:    "identical integers are equal",
	}, {
		label:     label + "/UnfilteredIgnore",
		x:         1,
		y:         1,
		opts:      []cmp.Option{cmp.Ignore()},
		wantPanic: "cannot use an unfiltered option",
		reason:    "unfiltered options are functionally useless",
	}, {
		label:     label + "/UnfilteredCompare",
		x:         1,
		y:         1,
		opts:      []cmp.Option{cmp.Comparer(func(_, _ interface{}) bool { return true })},
		wantPanic: "cannot use an unfiltered option",
		reason:    "unfiltered options are functionally useless",
	}, {
		label:     label + "/UnfilteredTransform",
		x:         1,
		y:         1,
		opts:      []cmp.Option{cmp.Transformer("λ", func(x interface{}) interface{} { return x })},
		wantPanic: "cannot use an unfiltered option",
		reason:    "unfiltered options are functionally useless",
	}, {
		label: label + "/AmbiguousOptions",
		x:     1,
		y:     1,
		opts: []cmp.Option{
			cmp.Comparer(func(x, y int) bool { return true }),
			cmp.Transformer("λ", func(x int) float64 { return float64(x) }),
		},
		wantPanic: "ambiguous set of applicable options",
		reason:    "both options apply on int, leading to ambiguity",
	}, {
		label: label + "/IgnorePrecedence",
		x:     1,
		y:     1,
		opts: []cmp.Option{
			cmp.FilterPath(func(p cmp.Path) bool {
				return len(p) > 0 && p[len(p)-1].Type().Kind() == reflect.Int
			}, cmp.Options{cmp.Ignore(), cmp.Ignore(), cmp.Ignore()}),
			cmp.Comparer(func(x, y int) bool { return true }),
			cmp.Transformer("λ", func(x int) float64 { return float64(x) }),
		},
		wantEqual: true,
		reason:    "ignore takes precedence over other options",
	}, {
		label:     label + "/UnknownOption",
		opts:      []cmp.Option{struct{ cmp.Option }{}},
		wantPanic: "unknown option",
		reason:    "use of unknown option should panic",
	}, {
		label:     label + "/StructEqual",
		x:         struct{ A, B, C int }{1, 2, 3},
		y:         struct{ A, B, C int }{1, 2, 3},
		wantEqual: true,
		reason:    "struct comparison with all equal fields",
	}, {
		label:     label + "/StructInequal",
		x:         struct{ A, B, C int }{1, 2, 3},
		y:         struct{ A, B, C int }{1, 2, 4},
		wantEqual: false,
		reason:    "struct comparison with inequal C field",
	}, {
		label:     label + "/StructUnexported",
		x:         struct{ a, b, c int }{1, 2, 3},
		y:         struct{ a, b, c int }{1, 2, 4},
		wantPanic: "cannot handle unexported field",
		reason:    "unexported fields result in a panic by default",
	}, {
		label:     label + "/PointerStructEqual",
		x:         &struct{ A *int }{newInt(4)},
		y:         &struct{ A *int }{newInt(4)},
		wantEqual: true,
		reason:    "comparison of pointer to struct with equal A field",
	}, {
		label:     label + "/PointerStructInequal",
		x:         &struct{ A *int }{newInt(4)},
		y:         &struct{ A *int }{newInt(5)},
		wantEqual: false,
		reason:    "comparison of pointer to struct with inequal A field",
	}, {
		label: label + "/PointerStructTrueComparer",
		x:     &struct{ A *int }{newInt(4)},
		y:     &struct{ A *int }{newInt(5)},
		opts: []cmp.Option{
			cmp.Comparer(func(x, y int) bool { return true }),
		},
		wantEqual: true,
		reason:    "comparison of pointer to struct with inequal A field, but treated as equal with always equal comparer",
	}, {
		label: label + "/PointerStructNonNilComparer",
		x:     &struct{ A *int }{newInt(4)},
		y:     &struct{ A *int }{newInt(5)},
		opts: []cmp.Option{
			cmp.Comparer(func(x, y *int) bool { return x != nil && y != nil }),
		},
		wantEqual: true,
		reason:    "comparison of pointer to struct with inequal A field, but treated as equal with comparer checking pointers for nilness",
	}, {
		label:     label + "/StructNestedPointerEqual",
		x:         &struct{ R *bytes.Buffer }{},
		y:         &struct{ R *bytes.Buffer }{},
		wantEqual: true,
		reason:    "equal since both pointers in R field are nil",
	}, {
		label:     label + "/StructNestedPointerInequal",
		x:         &struct{ R *bytes.Buffer }{new(bytes.Buffer)},
		y:         &struct{ R *bytes.Buffer }{},
		wantEqual: false,
		reason:    "inequal since R field is inequal",
	}, {
		label: label + "/StructNestedPointerTrueComparer",
		x:     &struct{ R *bytes.Buffer }{new(bytes.Buffer)},
		y:     &struct{ R *bytes.Buffer }{},
		opts: []cmp.Option{
			cmp.Comparer(func(x, y io.Reader) bool { return true }),
		},
		wantEqual: true,
		reason:    "equal despite inequal R field values since the comparer always reports true",
	}, {
		label:     label + "/StructNestedValueUnexportedPanic1",
		x:         &struct{ R bytes.Buffer }{},
		y:         &struct{ R bytes.Buffer }{},
		wantPanic: "cannot handle unexported field",
		reason:    "bytes.Buffer contains unexported fields",
	}, {
		label: label + "/StructNestedValueUnexportedPanic2",
		x:     &struct{ R bytes.Buffer }{},
		y:     &struct{ R bytes.Buffer }{},
		opts: []cmp.Option{
			cmp.Comparer(func(x, y io.Reader) bool { return true }),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "bytes.Buffer value does not implement io.Reader",
	}, {
		label: label + "/StructNestedValueEqual",
		x:     &struct{ R bytes.Buffer }{},
		y:     &struct{ R bytes.Buffer }{},
		opts: []cmp.Option{
			cmp.Transformer("Ref", func(x bytes.Buffer) *bytes.Buffer { return &x }),
			cmp.Comparer(func(x, y io.Reader) bool { return true }),
		},
		wantEqual: true,
		reason:    "bytes.Buffer pointer due to shallow copy does implement io.Reader",
	}, {
		label:     label + "/RegexpUnexportedPanic",
		x:         []*regexp.Regexp{nil, regexp.MustCompile("a*b*c*")},
		y:         []*regexp.Regexp{nil, regexp.MustCompile("a*b*c*")},
		wantPanic: "cannot handle unexported field",
		reason:    "regexp.Regexp contains unexported fields",
	}, {
		label: label + "/RegexpEqual",
		x:     []*regexp.Regexp{nil, regexp.MustCompile("a*b*c*")},
		y:     []*regexp.Regexp{nil, regexp.MustCompile("a*b*c*")},
		opts: []cmp.Option{cmp.Comparer(func(x, y *regexp.Regexp) bool {
			if x == nil || y == nil {
				return x == nil && y == nil
			}
			return x.String() == y.String()
		})},
		wantEqual: true,
		reason:    "comparer for *regexp.Regexp applied with equal regexp strings",
	}, {
		label: label + "/RegexpInequal",
		x:     []*regexp.Regexp{nil, regexp.MustCompile("a*b*c*")},
		y:     []*regexp.Regexp{nil, regexp.MustCompile("a*b*d*")},
		opts: []cmp.Option{cmp.Comparer(func(x, y *regexp.Regexp) bool {
			if x == nil || y == nil {
				return x == nil && y == nil
			}
			return x.String() == y.String()
		})},
		wantEqual: false,
		reason:    "comparer for *regexp.Regexp applied with inequal regexp strings",
	}, {
		label: label + "/TriplePointerEqual",
		x: func() ***int {
			a := 0
			b := &a
			c := &b
			return &c
		}(),
		y: func() ***int {
			a := 0
			b := &a
			c := &b
			return &c
		}(),
		wantEqual: true,
		reason:    "three layers of pointers to the same value",
	}, {
		label: label + "/TriplePointerInequal",
		x: func() ***int {
			a := 0
			b := &a
			c := &b
			return &c
		}(),
		y: func() ***int {
			a := 1
			b := &a
			c := &b
			return &c
		}(),
		wantEqual: false,
		reason:    "three layers of pointers to different values",
	}, {
		label:     label + "/SliceWithDifferingCapacity",
		x:         []int{1, 2, 3, 4, 5}[:3],
		y:         []int{1, 2, 3},
		wantEqual: true,
		reason:    "elements past the slice length are not compared",
	}, {
		label:     label + "/StringerEqual",
		x:         struct{ fmt.Stringer }{bytes.NewBufferString("hello")},
		y:         struct{ fmt.Stringer }{regexp.MustCompile("hello")},
		opts:      []cmp.Option{cmp.Comparer(func(x, y fmt.Stringer) bool { return x.String() == y.String() })},
		wantEqual: true,
		reason:    "comparer for fmt.Stringer used to compare differing types with same string",
	}, {
		label:     label + "/StringerInequal",
		x:         struct{ fmt.Stringer }{bytes.NewBufferString("hello")},
		y:         struct{ fmt.Stringer }{regexp.MustCompile("hello2")},
		opts:      []cmp.Option{cmp.Comparer(func(x, y fmt.Stringer) bool { return x.String() == y.String() })},
		wantEqual: false,
		reason:    "comparer for fmt.Stringer used to compare differing types with different strings",
	}, {
		label:     label + "/DifferingHash",
		x:         md5.Sum([]byte{'a'}),
		y:         md5.Sum([]byte{'b'}),
		wantEqual: false,
		reason:    "hash differs",
	}, {
		label:     label + "/NilStringer",
		x:         new(fmt.Stringer),
		y:         nil,
		wantEqual: false,
		reason:    "by default differing types are always inequal",
	}, {
		label:     label + "/TarHeaders",
		x:         makeTarHeaders('0'),
		y:         makeTarHeaders('\x00'),
		wantEqual: false,
		reason:    "type flag differs between the headers",
	}, {
		label: label + "/NonDeterministicComparer",
		x:     make([]int, 1000),
		y:     make([]int, 1000),
		opts: []cmp.Option{
			cmp.Comparer(func(_, _ int) bool {
				return rand.Intn(2) == 0
			}),
		},
		wantPanic: "non-deterministic or non-symmetric function detected",
		reason:    "non-deterministic comparer",
	}, {
		label: label + "/NonDeterministicFilter",
		x:     make([]int, 1000),
		y:     make([]int, 1000),
		opts: []cmp.Option{
			cmp.FilterValues(func(_, _ int) bool {
				return rand.Intn(2) == 0
			}, cmp.Ignore()),
		},
		wantPanic: "non-deterministic or non-symmetric function detected",
		reason:    "non-deterministic filter",
	}, {
		label: label + "/AssymetricComparer",
		x:     []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		y:     []int{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
		opts: []cmp.Option{
			cmp.Comparer(func(x, y int) bool {
				return x < y
			}),
		},
		wantPanic: "non-deterministic or non-symmetric function detected",
		reason:    "asymmetric comparer",
	}, {
		label: label + "/NonDeterministicTransformer",
		x:     make([]string, 1000),
		y:     make([]string, 1000),
		opts: []cmp.Option{
			cmp.Transformer("λ", func(x string) int {
				return rand.Int()
			}),
		},
		wantPanic: "non-deterministic function detected",
		reason:    "non-deterministic transformer",
	}, {
		label: label + "/IrreflexiveComparison",
		x:     make([]int, 10),
		y:     make([]int, 10),
		opts: []cmp.Option{
			cmp.Transformer("λ", func(x int) float64 {
				return math.NaN()
			}),
		},
		wantEqual: false,
		reason:    "dynamic checks should not panic for non-reflexive comparisons",
	}, {
		label:     label + "/StringerMapKey",
		x:         map[*pb.Stringer]*pb.Stringer{{"hello"}: {"world"}},
		y:         map[*pb.Stringer]*pb.Stringer(nil),
		wantEqual: false,
		reason:    "stringer should be used to format the map key",
	}, {
		label:     label + "/StringerBacktick",
		x:         []*pb.Stringer{{`multi\nline\nline\nline`}},
		wantEqual: false,
		reason:    "stringer should use backtick quoting if more readable",
	}, {
		label: label + "/AvoidPanicAssignableConverter",
		x:     struct{ I Iface2 }{},
		y:     struct{ I Iface2 }{},
		opts: []cmp.Option{
			cmp.Comparer(func(x, y Iface1) bool {
				return x == nil && y == nil
			}),
		},
		wantEqual: true,
		reason:    "function call using Go reflection should automatically convert assignable interfaces; see https://golang.org/issues/22143",
	}, {
		label: label + "/AvoidPanicAssignableTransformer",
		x:     struct{ I Iface2 }{},
		y:     struct{ I Iface2 }{},
		opts: []cmp.Option{
			cmp.Transformer("λ", func(v Iface1) bool {
				return v == nil
			}),
		},
		wantEqual: true,
		reason:    "function call using Go reflection should automatically convert assignable interfaces; see https://golang.org/issues/22143",
	}, {
		label: label + "/AvoidPanicAssignableFilter",
		x:     struct{ I Iface2 }{},
		y:     struct{ I Iface2 }{},
		opts: []cmp.Option{
			cmp.FilterValues(func(x, y Iface1) bool {
				return x == nil && y == nil
			}, cmp.Ignore()),
		},
		wantEqual: true,
		reason:    "function call using Go reflection should automatically convert assignable interfaces; see https://golang.org/issues/22143",
	}, {
		label:     label + "/DynamicMap",
		x:         []interface{}{map[string]interface{}{"avg": 0.278, "hr": 65, "name": "Mark McGwire"}, map[string]interface{}{"avg": 0.288, "hr": 63, "name": "Sammy Sosa"}},
		y:         []interface{}{map[string]interface{}{"avg": 0.278, "hr": 65.0, "name": "Mark McGwire"}, map[string]interface{}{"avg": 0.288, "hr": 63.0, "name": "Sammy Sosa"}},
		wantEqual: false,
		reason:    "dynamic map with differing types (but semantically equivalent values) should be inequal",
	}, {
		label: label + "/MapKeyPointer",
		x: map[*int]string{
			new(int): "hello",
		},
		y: map[*int]string{
			new(int): "world",
		},
		wantEqual: false,
		reason:    "map keys should use shallow (rather than deep) pointer comparison",
	}, {
		label: label + "/IgnoreSliceElements",
		x: [2][]int{
			{0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 7, 8, 0, 9, 0, 0},
			{0, 1, 0, 0, 0, 20},
		},
		y: [2][]int{
			{1, 2, 3, 0, 4, 5, 6, 7, 0, 8, 9, 0, 0, 0},
			{0, 0, 1, 2, 0, 0, 0},
		},
		opts: []cmp.Option{
			cmp.FilterPath(func(p cmp.Path) bool {
				vx, vy := p.Last().Values()
				if vx.IsValid() && vx.Kind() == reflect.Int && vx.Int() == 0 {
					return true
				}
				if vy.IsValid() && vy.Kind() == reflect.Int && vy.Int() == 0 {
					return true
				}
				return false
			}, cmp.Ignore()),
		},
		wantEqual: false,
		reason:    "all zero slice elements are ignored (even if missing)",
	}, {
		label: label + "/IgnoreMapEntries",
		x: [2]map[string]int{
			{"ignore1": 0, "ignore2": 0, "keep1": 1, "keep2": 2, "KEEP3": 3, "IGNORE3": 0},
			{"keep1": 1, "ignore1": 0},
		},
		y: [2]map[string]int{
			{"ignore1": 0, "ignore3": 0, "ignore4": 0, "keep1": 1, "keep2": 2, "KEEP3": 3},
			{"keep1": 1, "keep2": 2, "ignore2": 0},
		},
		opts: []cmp.Option{
			cmp.FilterPath(func(p cmp.Path) bool {
				vx, vy := p.Last().Values()
				if vx.IsValid() && vx.Kind() == reflect.Int && vx.Int() == 0 {
					return true
				}
				if vy.IsValid() && vy.Kind() == reflect.Int && vy.Int() == 0 {
					return true
				}
				return false
			}, cmp.Ignore()),
		},
		wantEqual: false,
		reason:    "all zero map entries are ignored (even if missing)",
	}, {
		label:     label + "/PanicUnexportedNamed",
		x:         namedWithUnexported{},
		y:         namedWithUnexported{},
		wantPanic: strconv.Quote(reflect.TypeOf(namedWithUnexported{}).PkgPath()) + ".namedWithUnexported",
		reason:    "panic on named struct type with unexported field",
	}, {
		label:     label + "/PanicUnexportedUnnamed",
		x:         struct{ a int }{},
		y:         struct{ a int }{},
		wantPanic: strconv.Quote(reflect.TypeOf(namedWithUnexported{}).PkgPath()) + ".(struct { a int })",
		reason:    "panic on unnamed struct type with unexported field",
	}, {
		label: label + "/UnaddressableStruct",
		x:     struct{ s fmt.Stringer }{new(bytes.Buffer)},
		y:     struct{ s fmt.Stringer }{nil},
		opts: []cmp.Option{
			cmp.AllowUnexported(struct{ s fmt.Stringer }{}),
			cmp.FilterPath(func(p cmp.Path) bool {
				if _, ok := p.Last().(cmp.StructField); !ok {
					return false
				}

				t := p.Index(-1).Type()
				vx, vy := p.Index(-1).Values()
				pvx, pvy := p.Index(-2).Values()
				switch {
				case vx.Type() != t:
					panic(fmt.Sprintf("inconsistent type: %v != %v", vx.Type(), t))
				case vy.Type() != t:
					panic(fmt.Sprintf("inconsistent type: %v != %v", vy.Type(), t))
				case vx.CanAddr() != pvx.CanAddr():
					panic(fmt.Sprintf("inconsistent addressability: %v != %v", vx.CanAddr(), pvx.CanAddr()))
				case vy.CanAddr() != pvy.CanAddr():
					panic(fmt.Sprintf("inconsistent addressability: %v != %v", vy.CanAddr(), pvy.CanAddr()))
				}
				return true
			}, cmp.Ignore()),
		},
		wantEqual: true,
		reason:    "verify that exporter does not leak implementation details",
	}, {
		label:     label + "/ErrorPanic",
		x:         io.EOF,
		y:         io.EOF,
		wantPanic: "consider using cmpopts.EquateErrors",
		reason:    "suggest cmpopts.EquateErrors when accessing unexported fields of error types",
	}, {
		label:     label + "/ErrorEqual",
		x:         io.EOF,
		y:         io.EOF,
		opts:      []cmp.Option{cmpopts.EquateErrors()},
		wantEqual: true,
		reason:    "cmpopts.EquateErrors should equate these two errors as sentinel values",
	}}
}

func transformerTests() []test {
	type StringBytes struct {
		String string
		Bytes  []byte
	}

	const label = "Transformer"

	transformOnce := func(name string, f interface{}) cmp.Option {
		xform := cmp.Transformer(name, f)
		return cmp.FilterPath(func(p cmp.Path) bool {
			for _, ps := range p {
				if tr, ok := ps.(cmp.Transform); ok && tr.Option() == xform {
					return false
				}
			}
			return true
		}, xform)
	}

	return []test{{
		label: label + "/Uints",
		x:     uint8(0),
		y:     uint8(1),
		opts: []cmp.Option{
			cmp.Transformer("λ", func(in uint8) uint16 { return uint16(in) }),
			cmp.Transformer("λ", func(in uint16) uint32 { return uint32(in) }),
			cmp.Transformer("λ", func(in uint32) uint64 { return uint64(in) }),
		},
		wantEqual: false,
		reason:    "transform uint8 -> uint16 -> uint32 -> uint64",
	}, {
		label: label + "/Ambiguous",
		x:     0,
		y:     1,
		opts: []cmp.Option{
			cmp.Transformer("λ", func(in int) int { return in / 2 }),
			cmp.Transformer("λ", func(in int) int { return in }),
		},
		wantPanic: "ambiguous set of applicable options",
		reason:    "both transformers apply on int",
	}, {
		label: label + "/Filtered",
		x:     []int{0, -5, 0, -1},
		y:     []int{1, 3, 0, -5},
		opts: []cmp.Option{
			cmp.FilterValues(
				func(x, y int) bool { return x+y >= 0 },
				cmp.Transformer("λ", func(in int) int64 { return int64(in / 2) }),
			),
			cmp.FilterValues(
				func(x, y int) bool { return x+y < 0 },
				cmp.Transformer("λ", func(in int) int64 { return int64(in) }),
			),
		},
		wantEqual: false,
		reason:    "disjoint transformers filtered based on the values",
	}, {
		label: label + "/DisjointOutput",
		x:     0,
		y:     1,
		opts: []cmp.Option{
			cmp.Transformer("λ", func(in int) interface{} {
				if in == 0 {
					return "zero"
				}
				return float64(in)
			}),
		},
		wantEqual: false,
		reason:    "output type differs based on input value",
	}, {
		label: label + "/JSON",
		x: `{
		  "firstName": "John",
		  "lastName": "Smith",
		  "age": 25,
		  "isAlive": true,
		  "address": {
		    "city": "Los Angeles",
		    "postalCode": "10021-3100",
		    "state": "CA",
		    "streetAddress": "21 2nd Street"
		  },
		  "phoneNumbers": [{
		    "type": "home",
		    "number": "212 555-4321"
		  },{
		    "type": "office",
		    "number": "646 555-4567"
		  },{
		    "number": "123 456-7890",
		    "type": "mobile"
		  }],
		  "children": []
		}`,
		y: `{"firstName":"John","lastName":"Smith","isAlive":true,"age":25,
			"address":{"streetAddress":"21 2nd Street","city":"New York",
			"state":"NY","postalCode":"10021-3100"},"phoneNumbers":[{"type":"home",
			"number":"212 555-1234"},{"type":"office","number":"646 555-4567"},{
			"type":"mobile","number":"123 456-7890"}],"children":[],"spouse":null}`,
		opts: []cmp.Option{
			transformOnce("ParseJSON", func(s string) (m map[string]interface{}) {
				if err := json.Unmarshal([]byte(s), &m); err != nil {
					panic(err)
				}
				return m
			}),
		},
		wantEqual: false,
		reason:    "transformer used to parse JSON input",
	}, {
		label: label + "/AcyclicString",
		x:     StringBytes{String: "some\nmulti\nLine\nstring", Bytes: []byte("some\nmulti\nline\nbytes")},
		y:     StringBytes{String: "some\nmulti\nline\nstring", Bytes: []byte("some\nmulti\nline\nBytes")},
		opts: []cmp.Option{
			transformOnce("SplitString", func(s string) []string { return strings.Split(s, "\n") }),
			transformOnce("SplitBytes", func(b []byte) [][]byte { return bytes.Split(b, []byte("\n")) }),
		},
		wantEqual: false,
		reason:    "string -> []string and []byte -> [][]byte transformer only applied once",
	}, {
		label: label + "/CyclicString",
		x:     "a\nb\nc\n",
		y:     "a\nb\nc\n",
		opts: []cmp.Option{
			cmp.Transformer("SplitLines", func(s string) []string { return strings.Split(s, "\n") }),
		},
		wantPanic: "recursive set of Transformers detected",
		reason:    "cyclic transformation from string -> []string -> string",
	}, {
		label: label + "/CyclicComplex",
		x:     complex64(0),
		y:     complex64(0),
		opts: []cmp.Option{
			cmp.Transformer("T1", func(x complex64) complex128 { return complex128(x) }),
			cmp.Transformer("T2", func(x complex128) [2]float64 { return [2]float64{real(x), imag(x)} }),
			cmp.Transformer("T3", func(x float64) complex64 { return complex64(complex(x, 0)) }),
		},
		wantPanic: "recursive set of Transformers detected",
		reason:    "cyclic transformation from complex64 -> complex128 -> [2]float64 -> complex64",
	}}
}

func reporterTests() []test {
	const label = "Reporter"

	type (
		MyString    string
		MyByte      byte
		MyBytes     []byte
		MyInt       int8
		MyInts      []int8
		MyUint      int16
		MyUints     []int16
		MyFloat     float32
		MyFloats    []float32
		MyComposite struct {
			StringA string
			StringB MyString
			BytesA  []byte
			BytesB  []MyByte
			BytesC  MyBytes
			IntsA   []int8
			IntsB   []MyInt
			IntsC   MyInts
			UintsA  []uint16
			UintsB  []MyUint
			UintsC  MyUints
			FloatsA []float32
			FloatsB []MyFloat
			FloatsC MyFloats
		}
	)

	return []test{{
		label:     label + "/PanicStringer",
		x:         struct{ X fmt.Stringer }{struct{ fmt.Stringer }{nil}},
		y:         struct{ X fmt.Stringer }{bytes.NewBuffer(nil)},
		wantEqual: false,
		reason:    "panic from fmt.Stringer should not crash the reporter",
	}, {
		label:     label + "/PanicError",
		x:         struct{ X error }{struct{ error }{nil}},
		y:         struct{ X error }{errors.New("")},
		wantEqual: false,
		reason:    "panic from error should not crash the reporter",
	}, {
		label:     label + "/AmbiguousType",
		x:         foo1.Bar{},
		y:         foo2.Bar{},
		wantEqual: false,
		reason:    "reporter should display the qualified type name to disambiguate between the two values",
	}, {
		label: label + "/AmbiguousPointer",
		x:     newInt(0),
		y:     newInt(0),
		opts: []cmp.Option{
			cmp.Comparer(func(x, y *int) bool { return x == y }),
		},
		wantEqual: false,
		reason:    "reporter should display the address to disambiguate between the two values",
	}, {
		label: label + "/AmbiguousPointerStruct",
		x:     struct{ I *int }{newInt(0)},
		y:     struct{ I *int }{newInt(0)},
		opts: []cmp.Option{
			cmp.Comparer(func(x, y *int) bool { return x == y }),
		},
		wantEqual: false,
		reason:    "reporter should display the address to disambiguate between the two struct fields",
	}, {
		label: label + "/AmbiguousPointerSlice",
		x:     []*int{newInt(0)},
		y:     []*int{newInt(0)},
		opts: []cmp.Option{
			cmp.Comparer(func(x, y *int) bool { return x == y }),
		},
		wantEqual: false,
		reason:    "reporter should display the address to disambiguate between the two slice elements",
	}, {
		label: label + "/AmbiguousPointerMap",
		x:     map[string]*int{"zero": newInt(0)},
		y:     map[string]*int{"zero": newInt(0)},
		opts: []cmp.Option{
			cmp.Comparer(func(x, y *int) bool { return x == y }),
		},
		wantEqual: false,
		reason:    "reporter should display the address to disambiguate between the two map values",
	}, {
		label:     label + "/AmbiguousStringer",
		x:         Stringer("hello"),
		y:         newStringer("hello"),
		wantEqual: false,
		reason:    "reporter should avoid calling String to disambiguate between the two values",
	}, {
		label:     label + "/AmbiguousStringerStruct",
		x:         struct{ S fmt.Stringer }{Stringer("hello")},
		y:         struct{ S fmt.Stringer }{newStringer("hello")},
		wantEqual: false,
		reason:    "reporter should avoid calling String to disambiguate between the two struct fields",
	}, {
		label:     label + "/AmbiguousStringerSlice",
		x:         []fmt.Stringer{Stringer("hello")},
		y:         []fmt.Stringer{newStringer("hello")},
		wantEqual: false,
		reason:    "reporter should avoid calling String to disambiguate between the two slice elements",
	}, {
		label:     label + "/AmbiguousStringerMap",
		x:         map[string]fmt.Stringer{"zero": Stringer("hello")},
		y:         map[string]fmt.Stringer{"zero": newStringer("hello")},
		wantEqual: false,
		reason:    "reporter should avoid calling String to disambiguate between the two map values",
	}, {
		label: label + "/AmbiguousSliceHeader",
		x:     make([]int, 0, 5),
		y:     make([]int, 0, 1000),
		opts: []cmp.Option{
			cmp.Comparer(func(x, y []int) bool { return cap(x) == cap(y) }),
		},
		wantEqual: false,
		reason:    "reporter should display the slice header to disambiguate between the two slice values",
	}, {
		label: label + "/AmbiguousStringerMapKey",
		x: map[interface{}]string{
			nil:               "nil",
			Stringer("hello"): "goodbye",
			foo1.Bar{"fizz"}:  "buzz",
		},
		y: map[interface{}]string{
			newStringer("hello"): "goodbye",
			foo2.Bar{"fizz"}:     "buzz",
		},
		wantEqual: false,
		reason:    "reporter should avoid calling String to disambiguate between the two map keys",
	}, {
		label:     label + "/NonAmbiguousStringerMapKey",
		x:         map[interface{}]string{Stringer("hello"): "goodbye"},
		y:         map[interface{}]string{newStringer("fizz"): "buzz"},
		wantEqual: false,
		reason:    "reporter should call String as there is no ambiguity between the two map keys",
	}, {
		label:     label + "/InvalidUTF8",
		x:         MyString("\xed\xa0\x80"),
		wantEqual: false,
		reason:    "invalid UTF-8 should format as quoted string",
	}, {
		label:     label + "/UnbatchedSlice",
		x:         MyComposite{IntsA: []int8{11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}},
		y:         MyComposite{IntsA: []int8{10, 11, 21, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}},
		wantEqual: false,
		reason:    "unbatched diffing desired since few elements differ",
	}, {
		label:     label + "/BatchedSlice",
		x:         MyComposite{IntsA: []int8{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}},
		y:         MyComposite{IntsA: []int8{12, 29, 13, 27, 22, 23, 17, 18, 19, 20, 21, 10, 26, 16, 25, 28, 11, 15, 24, 14}},
		wantEqual: false,
		reason:    "batched diffing desired since many elements differ",
	}, {
		label:     label + "/BatchedWithComparer",
		x:         MyComposite{BytesA: []byte{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}},
		y:         MyComposite{BytesA: []byte{12, 29, 13, 27, 22, 23, 17, 18, 19, 20, 21, 10, 26, 16, 25, 28, 11, 15, 24, 14}},
		wantEqual: false,
		opts: []cmp.Option{
			cmp.Comparer(bytes.Equal),
		},
		reason: "batched diffing desired since many elements differ",
	}, {
		label:     label + "/BatchedLong",
		x:         MyComposite{IntsA: []int8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127}},
		wantEqual: false,
		reason:    "batched output desired for a single slice of primitives unique to one of the inputs",
	}, {
		label: label + "/BatchedNamedAndUnnamed",
		x: MyComposite{
			BytesA:  []byte{1, 2, 3},
			BytesB:  []MyByte{4, 5, 6},
			BytesC:  MyBytes{7, 8, 9},
			IntsA:   []int8{-1, -2, -3},
			IntsB:   []MyInt{-4, -5, -6},
			IntsC:   MyInts{-7, -8, -9},
			UintsA:  []uint16{1000, 2000, 3000},
			UintsB:  []MyUint{4000, 5000, 6000},
			UintsC:  MyUints{7000, 8000, 9000},
			FloatsA: []float32{1.5, 2.5, 3.5},
			FloatsB: []MyFloat{4.5, 5.5, 6.5},
			FloatsC: MyFloats{7.5, 8.5, 9.5},
		},
		y: MyComposite{
			BytesA:  []byte{3, 2, 1},
			BytesB:  []MyByte{6, 5, 4},
			BytesC:  MyBytes{9, 8, 7},
			IntsA:   []int8{-3, -2, -1},
			IntsB:   []MyInt{-6, -5, -4},
			IntsC:   MyInts{-9, -8, -7},
			UintsA:  []uint16{3000, 2000, 1000},
			UintsB:  []MyUint{6000, 5000, 4000},
			UintsC:  MyUints{9000, 8000, 7000},
			FloatsA: []float32{3.5, 2.5, 1.5},
			FloatsB: []MyFloat{6.5, 5.5, 4.5},
			FloatsC: MyFloats{9.5, 8.5, 7.5},
		},
		wantEqual: false,
		reason:    "batched diffing available for both named and unnamed slices",
	}, {
		label:     label + "/BinaryHexdump",
		x:         MyComposite{BytesA: []byte("\xf3\x0f\x8a\xa4\xd3\x12R\t$\xbeX\x95A\xfd$fX\x8byT\xac\r\xd8qwp\x20j\\s\u007f\x8c\x17U\xc04\xcen\xf7\xaaG\xee2\x9d\xc5\xca\x1eX\xaf\x8f'\xf3\x02J\x90\xedi.p2\xb4\xab0 \xb6\xbd\\b4\x17\xb0\x00\xbbO~'G\x06\xf4.f\xfdc\xd7\x04ݷ0\xb7\xd1U~{\xf6\xb3~\x1dWi \x9e\xbc\xdf\xe1M\xa9\xef\xa2\xd2\xed\xb4Gx\xc9\xc9'\xa4\xc6\xce\xecDp]")},
		y:         MyComposite{BytesA: []byte("\xf3\x0f\x8a\xa4\xd3\x12R\t$\xbeT\xac\r\xd8qwp\x20j\\s\u007f\x8c\x17U\xc04\xcen\xf7\xaaG\xee2\x9d\xc5\xca\x1eX\xaf\x8f'\xf3\x02J\x90\xedi.p2\xb4\xab0 \xb6\xbd\\b4\x17\xb0\x00\xbbO~'G\x06\xf4.f\xfdc\xd7\x04ݷ0\xb7\xd1u-[]]\xf6\xb3haha~\x1dWI \x9e\xbc\xdf\xe1M\xa9\xef\xa2\xd2\xed\xb4Gx\xc9\xc9'\xa4\xc6\xce\xecDp]")},
		wantEqual: false,
		reason:    "binary diff in hexdump form since data is binary data",
	}, {
		label:     label + "/StringHexdump",
		x:         MyComposite{StringB: MyString("readme.txt\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000000600\x000000000\x000000000\x0000000000046\x0000000000000\x00011173\x00 0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00ustar\x0000\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000000000\x000000000\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")},
		y:         MyComposite{StringB: MyString("gopher.txt\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000000600\x000000000\x000000000\x0000000000043\x0000000000000\x00011217\x00 0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00ustar\x0000\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000000000\x000000000\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")},
		wantEqual: false,
		reason:    "binary diff desired since string looks like binary data",
	}, {
		label:     label + "/BinaryString",
		x:         MyComposite{BytesA: []byte(`{"firstName":"John","lastName":"Smith","isAlive":true,"age":27,"address":{"streetAddress":"314 54th Avenue","city":"New York","state":"NY","postalCode":"10021-3100"},"phoneNumbers":[{"type":"home","number":"212 555-1234"},{"type":"office","number":"646 555-4567"},{"type":"mobile","number":"123 456-7890"}],"children":[],"spouse":null}`)},
		y:         MyComposite{BytesA: []byte(`{"firstName":"John","lastName":"Smith","isAlive":true,"age":27,"address":{"streetAddress":"21 2nd Street","city":"New York","state":"NY","postalCode":"10021-3100"},"phoneNumbers":[{"type":"home","number":"212 555-1234"},{"type":"office","number":"646 555-4567"},{"type":"mobile","number":"123 456-7890"}],"children":[],"spouse":null}`)},
		wantEqual: false,
		reason:    "batched textual diff desired since bytes looks like textual data",
	}, {
		label:     label + "/TripleQuote",
		x:         MyComposite{StringA: "aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n"},
		y:         MyComposite{StringA: "aaa\nbbb\nCCC\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nrrr\nSSS\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n"},
		wantEqual: false,
		reason:    "use triple-quote syntax",
	}, {
		label: label + "/TripleQuoteSlice",
		x: []string{
			"aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
			"aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		},
		y: []string{
			"aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\n",
			"aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		},
		wantEqual: false,
		reason:    "use triple-quote syntax for slices of strings",
	}, {
		label: label + "/TripleQuoteNamedTypes",
		x: MyComposite{
			StringB: MyString("aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz"),
			BytesC:  MyBytes("aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz"),
		},
		y: MyComposite{
			StringB: MyString("aaa\nbbb\nCCC\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nrrr\nSSS\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz"),
			BytesC:  MyBytes("aaa\nbbb\nCCC\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nrrr\nSSS\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz"),
		},
		wantEqual: false,
		reason:    "use triple-quote syntax for named types",
	}, {
		label: label + "/TripleQuoteSliceNamedTypes",
		x: []MyString{
			"aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
			"aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		},
		y: []MyString{
			"aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\n",
			"aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		},
		wantEqual: false,
		reason:    "use triple-quote syntax for slices of named strings",
	}, {
		label:     label + "/TripleQuoteEndlines",
		x:         "aaa\nbbb\nccc\nddd\neee\nfff\nggg\r\nhhh\n\riii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n\r",
		y:         "aaa\nbbb\nCCC\nddd\neee\nfff\nggg\r\nhhh\n\riii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nrrr\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz",
		wantEqual: false,
		reason:    "use triple-quote syntax",
	}, {
		label:     label + "/AvoidTripleQuoteAmbiguousQuotes",
		x:         "aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		y:         "aaa\nbbb\nCCC\nddd\neee\n\"\"\"\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nrrr\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		wantEqual: false,
		reason:    "avoid triple-quote syntax due to presence of ambiguous triple quotes",
	}, {
		label:     label + "/AvoidTripleQuoteAmbiguousEllipsis",
		x:         "aaa\nbbb\nccc\n...\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		y:         "aaa\nbbb\nCCC\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nrrr\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		wantEqual: false,
		reason:    "avoid triple-quote syntax due to presence of ambiguous ellipsis",
	}, {
		label:     label + "/AvoidTripleQuoteNonPrintable",
		x:         "aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		y:         "aaa\nbbb\nCCC\nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\no\roo\nppp\nqqq\nrrr\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		wantEqual: false,
		reason:    "use triple-quote syntax",
	}, {
		label:     label + "/AvoidTripleQuoteIdenticalWhitespace",
		x:         "aaa\nbbb\nccc\n ddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nRRR\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		y:         "aaa\nbbb\nccc \nddd\neee\nfff\nggg\nhhh\niii\njjj\nkkk\nlll\nmmm\nnnn\nooo\nppp\nqqq\nrrr\nsss\nttt\nuuu\nvvv\nwww\nxxx\nyyy\nzzz\n",
		wantEqual: false,
		reason:    "avoid triple-quote syntax due to visual equivalence of differences",
	}, {
		label: label + "/TripleQuoteStringer",
		x: []fmt.Stringer{
			bytes.NewBuffer([]byte("package main\n\nimport (\n\t\"fmt\"\n)\n\nfunc main() {\n\tfmt.Println(\"Hello, playground\")\n}\n")),
			bytes.NewBuffer([]byte("package main\n\nimport (\n\t\"fmt\"\n\t\"math/rand\"\n)\n\nfunc main() {\n\tfmt.Println(\"My favorite number is\", rand.Intn(10))\n}\n")),
		},
		y: []fmt.Stringer{
			bytes.NewBuffer([]byte("package main\n\nimport (\n\t\"fmt\"\n)\n\nfunc main() {\n\tfmt.Println(\"Hello, playground\")\n}\n")),
			bytes.NewBuffer([]byte("package main\n\nimport (\n\t\"fmt\"\n\t\"math\"\n)\n\nfunc main() {\n\tfmt.Printf(\"Now you have %g problems.\\n\", math.Sqrt(7))\n}\n")),
		},
		opts:      []cmp.Option{cmp.Comparer(func(x, y fmt.Stringer) bool { return x.String() == y.String() })},
		wantEqual: false,
		reason:    "multi-line String output should be formatted with triple quote",
	}, {
		label:     label + "/LimitMaximumBytesDiffs",
		x:         []byte("\xcd====\x06\x1f\xc2\xcc\xc2-S=====\x1d\xdfa\xae\x98\x9fH======ǰ\xb7=======\xef====:\\\x94\xe6J\xc7=====\xb4======\n\n\xf7\x94===========\xf2\x9c\xc0f=====4\xf6\xf1\xc3\x17\x82======n\x16`\x91D\xc6\x06=======\x1cE====.===========\xc4\x18=======\x8a\x8d\x0e====\x87\xb1\xa5\x8e\xc3=====z\x0f1\xaeU======G,=======5\xe75\xee\x82\xf4\xce====\x11r===========\xaf]=======z\x05\xb3\x91\x88%\xd2====\n1\x89=====i\xb7\x055\xe6\x81\xd2=============\x883=@̾====\x14\x05\x96%^t\x04=====\xe7Ȉ\x90\x1d============="),
		y:         []byte("\\====|\x96\xe7SB\xa0\xab=====\xf0\xbd\xa5q\xab\x17;======\xabP\x00=======\xeb====\xa5\x14\xe6O(\xe4=====(======/c@?===========\xd9x\xed\x13=====J\xfc\x918B\x8d======a8A\xebs\x04\xae=======\aC====\x1c===========\x91\"=======uؾ====s\xec\x845\a=====;\xabS9t======\x1f\x1b=======\x80\xab/\xed+:;====\xeaI===========\xabl=======\xb9\xe9\xfdH\x93\x8e\u007f====ח\xe5=====Ig\x88m\xf5\x01V=============\xf7+4\xb0\x92E====\x9fj\xf8&\xd0h\xf9=====\xeeΨ\r\xbf============="),
		wantEqual: false,
		reason:    "total bytes difference output is truncated due to excessive number of differences",
	}, {
		label:     label + "/LimitMaximumStringDiffs",
		x:         "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\nA\nB\nC\nD\nE\nF\nG\nH\nI\nJ\nK\nL\nM\nN\nO\nP\nQ\nR\nS\nT\nU\nV\nW\nX\nY\nZ\n",
		y:         "aa\nb\ncc\nd\nee\nf\ngg\nh\nii\nj\nkk\nl\nmm\nn\noo\np\nqq\nr\nss\nt\nuu\nv\nww\nx\nyy\nz\nAA\nB\nCC\nD\nEE\nF\nGG\nH\nII\nJ\nKK\nL\nMM\nN\nOO\nP\nQQ\nR\nSS\nT\nUU\nV\nWW\nX\nYY\nZ\n",
		wantEqual: false,
		reason:    "total string difference output is truncated due to excessive number of differences",
	}, {
		label: label + "/LimitMaximumSliceDiffs",
		x: func() (out []struct{ S string }) {
			for _, s := range strings.Split("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\nA\nB\nC\nD\nE\nF\nG\nH\nI\nJ\nK\nL\nM\nN\nO\nP\nQ\nR\nS\nT\nU\nV\nW\nX\nY\nZ\n", "\n") {
				out = append(out, struct{ S string }{s})
			}
			return out
		}(),
		y: func() (out []struct{ S string }) {
			for _, s := range strings.Split("aa\nb\ncc\nd\nee\nf\ngg\nh\nii\nj\nkk\nl\nmm\nn\noo\np\nqq\nr\nss\nt\nuu\nv\nww\nx\nyy\nz\nAA\nB\nCC\nD\nEE\nF\nGG\nH\nII\nJ\nKK\nL\nMM\nN\nOO\nP\nQQ\nR\nSS\nT\nUU\nV\nWW\nX\nYY\nZ\n", "\n") {
				out = append(out, struct{ S string }{s})
			}
			return out
		}(),
		wantEqual: false,
		reason:    "total slice difference output is truncated due to excessive number of differences",
	}, {
		label: label + "/MultilineString",
		x: MyComposite{
			StringA: strings.TrimPrefix(`
Package cmp determines equality of values.

This package is intended to be a more powerful and safer alternative to
reflect.DeepEqual for comparing whether two values are semantically equal.

The primary features of cmp are:

• When the default behavior of equality does not suit the needs of the test,
custom equality functions can override the equality operation.
For example, an equality function may report floats as equal so long as they
are within some tolerance of each other.

• Types that have an Equal method may use that method to determine equality.
This allows package authors to determine the equality operation for the types
that they define.

• If no custom equality functions are used and no Equal method is defined,
equality is determined by recursively comparing the primitive kinds on both
values, much like reflect.DeepEqual. Unlike reflect.DeepEqual, unexported
fields are not compared by default; they result in panics unless suppressed
by using an Ignore option (see cmpopts.IgnoreUnexported) or explicitly compared
using the AllowUnexported option.
`, "\n"),
		},
		y: MyComposite{
			StringA: strings.TrimPrefix(`
Package cmp determines equality of value.

This package is intended to be a more powerful and safer alternative to
reflect.DeepEqual for comparing whether two values are semantically equal.

The primary features of cmp are:

• When the default behavior of equality does not suit the needs of the test,
custom equality functions can override the equality operation.
For example, an equality function may report floats as equal so long as they
are within some tolerance of each other.

• If no custom equality functions are used and no Equal method is defined,
equality is determined by recursively comparing the primitive kinds on both
values, much like reflect.DeepEqual. Unlike reflect.DeepEqual, unexported
fields are not compared by default; they result in panics unless suppressed
by using an Ignore option (see cmpopts.IgnoreUnexported) or explicitly compared
using the AllowUnexported option.`, "\n"),
		},
		wantEqual: false,
		reason:    "batched per-line diff desired since string looks like multi-line textual data",
	}, {
		label: label + "/Slices",
		x: MyComposite{
			BytesA:  []byte{1, 2, 3},
			BytesB:  []MyByte{4, 5, 6},
			BytesC:  MyBytes{7, 8, 9},
			IntsA:   []int8{-1, -2, -3},
			IntsB:   []MyInt{-4, -5, -6},
			IntsC:   MyInts{-7, -8, -9},
			UintsA:  []uint16{1000, 2000, 3000},
			UintsB:  []MyUint{4000, 5000, 6000},
			UintsC:  MyUints{7000, 8000, 9000},
			FloatsA: []float32{1.5, 2.5, 3.5},
			FloatsB: []MyFloat{4.5, 5.5, 6.5},
			FloatsC: MyFloats{7.5, 8.5, 9.5},
		},
		y:         MyComposite{},
		wantEqual: false,
		reason:    "batched diffing for non-nil slices and nil slices",
	}, {
		label: label + "/EmptySlices",
		x: MyComposite{
			BytesA:  []byte{},
			BytesB:  []MyByte{},
			BytesC:  MyBytes{},
			IntsA:   []int8{},
			IntsB:   []MyInt{},
			IntsC:   MyInts{},
			UintsA:  []uint16{},
			UintsB:  []MyUint{},
			UintsC:  MyUints{},
			FloatsA: []float32{},
			FloatsB: []MyFloat{},
			FloatsC: MyFloats{},
		},
		y:         MyComposite{},
		wantEqual: false,
		reason:    "batched diffing for empty slices and nil slices",
	}, {
		label: label + "/LargeMapKey",
		x: map[*[]byte]int{func() *[]byte {
			b := make([]byte, 1<<20, 1<<20)
			return &b
		}(): 0},
		y: map[*[]byte]int{func() *[]byte {
			b := make([]byte, 1<<20, 1<<20)
			return &b
		}(): 0},
		reason: "printing map keys should have some verbosity limit imposed",
	}, {
		label:  label + "/LargeStringInInterface",
		x:      struct{ X interface{} }{"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam sit amet pretium ligula, at gravida quam. Integer iaculis, velit at sagittis ultricies, lacus metus scelerisque turpis, ornare feugiat nulla nisl ac erat. Maecenas elementum ultricies libero, sed efficitur lacus molestie non. Nulla ac pretium dolor. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Pellentesque mi lorem, consectetur id porttitor id, sollicitudin sit amet enim. Duis eu dolor magna. Nunc ut augue turpis."},
		y:      struct{ X interface{} }{"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam sit amet pretium ligula, at gravida quam. Integer iaculis, velit at sagittis ultricies, lacus metus scelerisque turpis, ornare feugiat nulla nisl ac erat. Maecenas elementum ultricies libero, sed efficitur lacus molestie non. Nulla ac pretium dolor. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Pellentesque mi lorem, consectetur id porttitor id, sollicitudin sit amet enim. Duis eu dolor magna. Nunc ut augue turpis,"},
		reason: "strings within an interface should benefit from specialized diffing",
	}, {
		label:  label + "/LargeBytesInInterface",
		x:      struct{ X interface{} }{[]byte("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam sit amet pretium ligula, at gravida quam. Integer iaculis, velit at sagittis ultricies, lacus metus scelerisque turpis, ornare feugiat nulla nisl ac erat. Maecenas elementum ultricies libero, sed efficitur lacus molestie non. Nulla ac pretium dolor. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Pellentesque mi lorem, consectetur id porttitor id, sollicitudin sit amet enim. Duis eu dolor magna. Nunc ut augue turpis.")},
		y:      struct{ X interface{} }{[]byte("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam sit amet pretium ligula, at gravida quam. Integer iaculis, velit at sagittis ultricies, lacus metus scelerisque turpis, ornare feugiat nulla nisl ac erat. Maecenas elementum ultricies libero, sed efficitur lacus molestie non. Nulla ac pretium dolor. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Pellentesque mi lorem, consectetur id porttitor id, sollicitudin sit amet enim. Duis eu dolor magna. Nunc ut augue turpis,")},
		reason: "bytes slice within an interface should benefit from specialized diffing",
	}, {
		label:  label + "/LargeStandaloneString",
		x:      struct{ X interface{} }{[1]string{"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam sit amet pretium ligula, at gravida quam. Integer iaculis, velit at sagittis ultricies, lacus metus scelerisque turpis, ornare feugiat nulla nisl ac erat. Maecenas elementum ultricies libero, sed efficitur lacus molestie non. Nulla ac pretium dolor. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Pellentesque mi lorem, consectetur id porttitor id, sollicitudin sit amet enim. Duis eu dolor magna. Nunc ut augue turpis."}},
		y:      struct{ X interface{} }{[1]string{"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam sit amet pretium ligula, at gravida quam. Integer iaculis, velit at sagittis ultricies, lacus metus scelerisque turpis, ornare feugiat nulla nisl ac erat. Maecenas elementum ultricies libero, sed efficitur lacus molestie non. Nulla ac pretium dolor. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Pellentesque mi lorem, consectetur id porttitor id, sollicitudin sit amet enim. Duis eu dolor magna. Nunc ut augue turpis,"}},
		reason: "printing a large standalone string that is different should print enough context to see the difference",
	}}
}

func embeddedTests() []test {
	const label = "EmbeddedStruct"

	privateStruct := *new(ts.ParentStructA).PrivateStruct()

	createStructA := func(i int) ts.ParentStructA {
		s := ts.ParentStructA{}
		s.PrivateStruct().Public = 1 + i
		s.PrivateStruct().SetPrivate(2 + i)
		return s
	}

	createStructB := func(i int) ts.ParentStructB {
		s := ts.ParentStructB{}
		s.PublicStruct.Public = 1 + i
		s.PublicStruct.SetPrivate(2 + i)
		return s
	}

	createStructC := func(i int) ts.ParentStructC {
		s := ts.ParentStructC{}
		s.PrivateStruct().Public = 1 + i
		s.PrivateStruct().SetPrivate(2 + i)
		s.Public = 3 + i
		s.SetPrivate(4 + i)
		return s
	}

	createStructD := func(i int) ts.ParentStructD {
		s := ts.ParentStructD{}
		s.PublicStruct.Public = 1 + i
		s.PublicStruct.SetPrivate(2 + i)
		s.Public = 3 + i
		s.SetPrivate(4 + i)
		return s
	}

	createStructE := func(i int) ts.ParentStructE {
		s := ts.ParentStructE{}
		s.PrivateStruct().Public = 1 + i
		s.PrivateStruct().SetPrivate(2 + i)
		s.PublicStruct.Public = 3 + i
		s.PublicStruct.SetPrivate(4 + i)
		return s
	}

	createStructF := func(i int) ts.ParentStructF {
		s := ts.ParentStructF{}
		s.PrivateStruct().Public = 1 + i
		s.PrivateStruct().SetPrivate(2 + i)
		s.PublicStruct.Public = 3 + i
		s.PublicStruct.SetPrivate(4 + i)
		s.Public = 5 + i
		s.SetPrivate(6 + i)
		return s
	}

	createStructG := func(i int) *ts.ParentStructG {
		s := ts.NewParentStructG()
		s.PrivateStruct().Public = 1 + i
		s.PrivateStruct().SetPrivate(2 + i)
		return s
	}

	createStructH := func(i int) *ts.ParentStructH {
		s := ts.NewParentStructH()
		s.PublicStruct.Public = 1 + i
		s.PublicStruct.SetPrivate(2 + i)
		return s
	}

	createStructI := func(i int) *ts.ParentStructI {
		s := ts.NewParentStructI()
		s.PrivateStruct().Public = 1 + i
		s.PrivateStruct().SetPrivate(2 + i)
		s.PublicStruct.Public = 3 + i
		s.PublicStruct.SetPrivate(4 + i)
		return s
	}

	createStructJ := func(i int) *ts.ParentStructJ {
		s := ts.NewParentStructJ()
		s.PrivateStruct().Public = 1 + i
		s.PrivateStruct().SetPrivate(2 + i)
		s.PublicStruct.Public = 3 + i
		s.PublicStruct.SetPrivate(4 + i)
		s.Private().Public = 5 + i
		s.Private().SetPrivate(6 + i)
		s.Public.Public = 7 + i
		s.Public.SetPrivate(8 + i)
		return s
	}

	// TODO(≥go1.10): Workaround for reflect bug (https://golang.org/issue/21122).
	wantPanicNotGo110 := func(s string) string {
		if !flags.AtLeastGo110 {
			return ""
		}
		return s
	}

	return []test{{
		label:     label + "/ParentStructA/PanicUnexported1",
		x:         ts.ParentStructA{},
		y:         ts.ParentStructA{},
		wantPanic: "cannot handle unexported field",
		reason:    "ParentStructA has an unexported field",
	}, {
		label: label + "/ParentStructA/Ignored",
		x:     ts.ParentStructA{},
		y:     ts.ParentStructA{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructA{}),
		},
		wantEqual: true,
		reason:    "the only field (which is unexported) of ParentStructA is ignored",
	}, {
		label: label + "/ParentStructA/PanicUnexported2",
		x:     createStructA(0),
		y:     createStructA(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructA{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "privateStruct also has unexported fields",
	}, {
		label: label + "/ParentStructA/Equal",
		x:     createStructA(0),
		y:     createStructA(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructA{}, privateStruct),
		},
		wantEqual: true,
		reason:    "unexported fields of both ParentStructA and privateStruct are allowed",
	}, {
		label: label + "/ParentStructA/Inequal",
		x:     createStructA(0),
		y:     createStructA(1),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructA{}, privateStruct),
		},
		wantEqual: false,
		reason:    "the two values differ on some fields",
	}, {
		label: label + "/ParentStructB/PanicUnexported1",
		x:     ts.ParentStructB{},
		y:     ts.ParentStructB{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructB{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "PublicStruct has an unexported field",
	}, {
		label: label + "/ParentStructB/Ignored",
		x:     ts.ParentStructB{},
		y:     ts.ParentStructB{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructB{}),
			cmpopts.IgnoreUnexported(ts.PublicStruct{}),
		},
		wantEqual: true,
		reason:    "unexported fields of both ParentStructB and PublicStruct are ignored",
	}, {
		label: label + "/ParentStructB/PanicUnexported2",
		x:     createStructB(0),
		y:     createStructB(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructB{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "PublicStruct also has unexported fields",
	}, {
		label: label + "/ParentStructB/Equal",
		x:     createStructB(0),
		y:     createStructB(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructB{}, ts.PublicStruct{}),
		},
		wantEqual: true,
		reason:    "unexported fields of both ParentStructB and PublicStruct are allowed",
	}, {
		label: label + "/ParentStructB/Inequal",
		x:     createStructB(0),
		y:     createStructB(1),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructB{}, ts.PublicStruct{}),
		},
		wantEqual: false,
		reason:    "the two values differ on some fields",
	}, {
		label:     label + "/ParentStructC/PanicUnexported1",
		x:         ts.ParentStructC{},
		y:         ts.ParentStructC{},
		wantPanic: "cannot handle unexported field",
		reason:    "ParentStructC has unexported fields",
	}, {
		label: label + "/ParentStructC/Ignored",
		x:     ts.ParentStructC{},
		y:     ts.ParentStructC{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructC{}),
		},
		wantEqual: true,
		reason:    "unexported fields of ParentStructC are ignored",
	}, {
		label: label + "/ParentStructC/PanicUnexported2",
		x:     createStructC(0),
		y:     createStructC(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructC{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "privateStruct also has unexported fields",
	}, {
		label: label + "/ParentStructC/Equal",
		x:     createStructC(0),
		y:     createStructC(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructC{}, privateStruct),
		},
		wantEqual: true,
		reason:    "unexported fields of both ParentStructC and privateStruct are allowed",
	}, {
		label: label + "/ParentStructC/Inequal",
		x:     createStructC(0),
		y:     createStructC(1),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructC{}, privateStruct),
		},
		wantEqual: false,
		reason:    "the two values differ on some fields",
	}, {
		label: label + "/ParentStructD/PanicUnexported1",
		x:     ts.ParentStructD{},
		y:     ts.ParentStructD{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructD{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "ParentStructD has unexported fields",
	}, {
		label: label + "/ParentStructD/Ignored",
		x:     ts.ParentStructD{},
		y:     ts.ParentStructD{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructD{}),
			cmpopts.IgnoreUnexported(ts.PublicStruct{}),
		},
		wantEqual: true,
		reason:    "unexported fields of ParentStructD and PublicStruct are ignored",
	}, {
		label: label + "/ParentStructD/PanicUnexported2",
		x:     createStructD(0),
		y:     createStructD(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructD{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "PublicStruct also has unexported fields",
	}, {
		label: label + "/ParentStructD/Equal",
		x:     createStructD(0),
		y:     createStructD(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructD{}, ts.PublicStruct{}),
		},
		wantEqual: true,
		reason:    "unexported fields of both ParentStructD and PublicStruct are allowed",
	}, {
		label: label + "/ParentStructD/Inequal",
		x:     createStructD(0),
		y:     createStructD(1),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructD{}, ts.PublicStruct{}),
		},
		wantEqual: false,
		reason:    "the two values differ on some fields",
	}, {
		label: label + "/ParentStructE/PanicUnexported1",
		x:     ts.ParentStructE{},
		y:     ts.ParentStructE{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructE{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "ParentStructE has unexported fields",
	}, {
		label: label + "/ParentStructE/Ignored",
		x:     ts.ParentStructE{},
		y:     ts.ParentStructE{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructE{}),
			cmpopts.IgnoreUnexported(ts.PublicStruct{}),
		},
		wantEqual: true,
		reason:    "unexported fields of ParentStructE and PublicStruct are ignored",
	}, {
		label: label + "/ParentStructE/PanicUnexported2",
		x:     createStructE(0),
		y:     createStructE(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructE{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "PublicStruct and privateStruct also has unexported fields",
	}, {
		label: label + "/ParentStructE/PanicUnexported3",
		x:     createStructE(0),
		y:     createStructE(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructE{}, ts.PublicStruct{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "privateStruct also has unexported fields",
	}, {
		label: label + "/ParentStructE/Equal",
		x:     createStructE(0),
		y:     createStructE(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructE{}, ts.PublicStruct{}, privateStruct),
		},
		wantEqual: true,
		reason:    "unexported fields of both ParentStructE, PublicStruct, and privateStruct are allowed",
	}, {
		label: label + "/ParentStructE/Inequal",
		x:     createStructE(0),
		y:     createStructE(1),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructE{}, ts.PublicStruct{}, privateStruct),
		},
		wantEqual: false,
		reason:    "the two values differ on some fields",
	}, {
		label: label + "/ParentStructF/PanicUnexported1",
		x:     ts.ParentStructF{},
		y:     ts.ParentStructF{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructF{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "ParentStructF has unexported fields",
	}, {
		label: label + "/ParentStructF/Ignored",
		x:     ts.ParentStructF{},
		y:     ts.ParentStructF{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructF{}),
			cmpopts.IgnoreUnexported(ts.PublicStruct{}),
		},
		wantEqual: true,
		reason:    "unexported fields of ParentStructF and PublicStruct are ignored",
	}, {
		label: label + "/ParentStructF/PanicUnexported2",
		x:     createStructF(0),
		y:     createStructF(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructF{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "PublicStruct and privateStruct also has unexported fields",
	}, {
		label: label + "/ParentStructF/PanicUnexported3",
		x:     createStructF(0),
		y:     createStructF(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructF{}, ts.PublicStruct{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "privateStruct also has unexported fields",
	}, {
		label: label + "/ParentStructF/Equal",
		x:     createStructF(0),
		y:     createStructF(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructF{}, ts.PublicStruct{}, privateStruct),
		},
		wantEqual: true,
		reason:    "unexported fields of both ParentStructF, PublicStruct, and privateStruct are allowed",
	}, {
		label: label + "/ParentStructF/Inequal",
		x:     createStructF(0),
		y:     createStructF(1),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructF{}, ts.PublicStruct{}, privateStruct),
		},
		wantEqual: false,
		reason:    "the two values differ on some fields",
	}, {
		label:     label + "/ParentStructG/PanicUnexported1",
		x:         ts.ParentStructG{},
		y:         ts.ParentStructG{},
		wantPanic: wantPanicNotGo110("cannot handle unexported field"),
		wantEqual: !flags.AtLeastGo110,
		reason:    "ParentStructG has unexported fields",
	}, {
		label: label + "/ParentStructG/Ignored",
		x:     ts.ParentStructG{},
		y:     ts.ParentStructG{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructG{}),
		},
		wantEqual: true,
		reason:    "unexported fields of ParentStructG are ignored",
	}, {
		label: label + "/ParentStructG/PanicUnexported2",
		x:     createStructG(0),
		y:     createStructG(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructG{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "privateStruct also has unexported fields",
	}, {
		label: label + "/ParentStructG/Equal",
		x:     createStructG(0),
		y:     createStructG(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructG{}, privateStruct),
		},
		wantEqual: true,
		reason:    "unexported fields of both ParentStructG and privateStruct are allowed",
	}, {
		label: label + "/ParentStructG/Inequal",
		x:     createStructG(0),
		y:     createStructG(1),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructG{}, privateStruct),
		},
		wantEqual: false,
		reason:    "the two values differ on some fields",
	}, {
		label:     label + "/ParentStructH/EqualNil",
		x:         ts.ParentStructH{},
		y:         ts.ParentStructH{},
		wantEqual: true,
		reason:    "PublicStruct is not compared because the pointer is nil",
	}, {
		label:     label + "/ParentStructH/PanicUnexported1",
		x:         createStructH(0),
		y:         createStructH(0),
		wantPanic: "cannot handle unexported field",
		reason:    "PublicStruct has unexported fields",
	}, {
		label: label + "/ParentStructH/Ignored",
		x:     ts.ParentStructH{},
		y:     ts.ParentStructH{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructH{}),
		},
		wantEqual: true,
		reason:    "unexported fields of ParentStructH are ignored (it has none)",
	}, {
		label: label + "/ParentStructH/PanicUnexported2",
		x:     createStructH(0),
		y:     createStructH(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructH{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "PublicStruct also has unexported fields",
	}, {
		label: label + "/ParentStructH/Equal",
		x:     createStructH(0),
		y:     createStructH(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructH{}, ts.PublicStruct{}),
		},
		wantEqual: true,
		reason:    "unexported fields of both ParentStructH and PublicStruct are allowed",
	}, {
		label: label + "/ParentStructH/Inequal",
		x:     createStructH(0),
		y:     createStructH(1),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructH{}, ts.PublicStruct{}),
		},
		wantEqual: false,
		reason:    "the two values differ on some fields",
	}, {
		label:     label + "/ParentStructI/PanicUnexported1",
		x:         ts.ParentStructI{},
		y:         ts.ParentStructI{},
		wantPanic: wantPanicNotGo110("cannot handle unexported field"),
		wantEqual: !flags.AtLeastGo110,
		reason:    "ParentStructI has unexported fields",
	}, {
		label: label + "/ParentStructI/Ignored1",
		x:     ts.ParentStructI{},
		y:     ts.ParentStructI{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructI{}),
		},
		wantEqual: true,
		reason:    "unexported fields of ParentStructI are ignored",
	}, {
		label: label + "/ParentStructI/PanicUnexported2",
		x:     createStructI(0),
		y:     createStructI(0),
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructI{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "PublicStruct and privateStruct also has unexported fields",
	}, {
		label: label + "/ParentStructI/Ignored2",
		x:     createStructI(0),
		y:     createStructI(0),
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructI{}, ts.PublicStruct{}),
		},
		wantEqual: true,
		reason:    "unexported fields of ParentStructI and PublicStruct are ignored",
	}, {
		label: label + "/ParentStructI/PanicUnexported3",
		x:     createStructI(0),
		y:     createStructI(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructI{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "PublicStruct and privateStruct also has unexported fields",
	}, {
		label: label + "/ParentStructI/Equal",
		x:     createStructI(0),
		y:     createStructI(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructI{}, ts.PublicStruct{}, privateStruct),
		},
		wantEqual: true,
		reason:    "unexported fields of both ParentStructI, PublicStruct, and privateStruct are allowed",
	}, {
		label: label + "/ParentStructI/Inequal",
		x:     createStructI(0),
		y:     createStructI(1),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructI{}, ts.PublicStruct{}, privateStruct),
		},
		wantEqual: false,
		reason:    "the two values differ on some fields",
	}, {
		label:     label + "/ParentStructJ/PanicUnexported1",
		x:         ts.ParentStructJ{},
		y:         ts.ParentStructJ{},
		wantPanic: "cannot handle unexported field",
		reason:    "ParentStructJ has unexported fields",
	}, {
		label: label + "/ParentStructJ/PanicUnexported2",
		x:     ts.ParentStructJ{},
		y:     ts.ParentStructJ{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructJ{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "PublicStruct and privateStruct also has unexported fields",
	}, {
		label: label + "/ParentStructJ/Ignored",
		x:     ts.ParentStructJ{},
		y:     ts.ParentStructJ{},
		opts: []cmp.Option{
			cmpopts.IgnoreUnexported(ts.ParentStructJ{}, ts.PublicStruct{}),
		},
		wantEqual: true,
		reason:    "unexported fields of ParentStructJ and PublicStruct are ignored",
	}, {
		label: label + "/ParentStructJ/PanicUnexported3",
		x:     createStructJ(0),
		y:     createStructJ(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructJ{}, ts.PublicStruct{}),
		},
		wantPanic: "cannot handle unexported field",
		reason:    "privateStruct also has unexported fields",
	}, {
		label: label + "/ParentStructJ/Equal",
		x:     createStructJ(0),
		y:     createStructJ(0),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructJ{}, ts.PublicStruct{}, privateStruct),
		},
		wantEqual: true,
		reason:    "unexported fields of both ParentStructJ, PublicStruct, and privateStruct are allowed",
	}, {
		label: label + "/ParentStructJ/Inequal",
		x:     createStructJ(0),
		y:     createStructJ(1),
		opts: []cmp.Option{
			cmp.AllowUnexported(ts.ParentStructJ{}, ts.PublicStruct{}, privateStruct),
		},
		wantEqual: false,
		reason:    "the two values differ on some fields",
	}}
}

func methodTests() []test {
	const label = "EqualMethod"

	// A common mistake that the Equal method is on a pointer receiver,
	// but only a non-pointer value is present in the struct.
	// A transform can be used to forcibly reference the value.
	addrTransform := cmp.FilterPath(func(p cmp.Path) bool {
		if len(p) == 0 {
			return false
		}
		t := p[len(p)-1].Type()
		if _, ok := t.MethodByName("Equal"); ok || t.Kind() == reflect.Ptr {
			return false
		}
		if m, ok := reflect.PtrTo(t).MethodByName("Equal"); ok {
			tf := m.Func.Type()
			return !tf.IsVariadic() && tf.NumIn() == 2 && tf.NumOut() == 1 &&
				tf.In(0).AssignableTo(tf.In(1)) && tf.Out(0) == reflect.TypeOf(true)
		}
		return false
	}, cmp.Transformer("Addr", func(x interface{}) interface{} {
		v := reflect.ValueOf(x)
		vp := reflect.New(v.Type())
		vp.Elem().Set(v)
		return vp.Interface()
	}))

	// For each of these types, there is an Equal method defined, which always
	// returns true, while the underlying data are fundamentally different.
	// Since the method should be called, these are expected to be equal.
	return []test{{
		label:     label + "/StructA/ValueEqual",
		x:         ts.StructA{X: "NotEqual"},
		y:         ts.StructA{X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructA value called",
	}, {
		label:     label + "/StructA/PointerEqual",
		x:         &ts.StructA{X: "NotEqual"},
		y:         &ts.StructA{X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructA pointer called",
	}, {
		label:     label + "/StructB/ValueInequal",
		x:         ts.StructB{X: "NotEqual"},
		y:         ts.StructB{X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructB value not called",
	}, {
		label:     label + "/StructB/ValueAddrEqual",
		x:         ts.StructB{X: "NotEqual"},
		y:         ts.StructB{X: "not_equal"},
		opts:      []cmp.Option{addrTransform},
		wantEqual: true,
		reason:    "Equal method on StructB pointer called due to shallow copy transform",
	}, {
		label:     label + "/StructB/PointerEqual",
		x:         &ts.StructB{X: "NotEqual"},
		y:         &ts.StructB{X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructB pointer called",
	}, {
		label:     label + "/StructC/ValueEqual",
		x:         ts.StructC{X: "NotEqual"},
		y:         ts.StructC{X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructC value called",
	}, {
		label:     label + "/StructC/PointerEqual",
		x:         &ts.StructC{X: "NotEqual"},
		y:         &ts.StructC{X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructC pointer called",
	}, {
		label:     label + "/StructD/ValueInequal",
		x:         ts.StructD{X: "NotEqual"},
		y:         ts.StructD{X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructD value not called",
	}, {
		label:     label + "/StructD/ValueAddrEqual",
		x:         ts.StructD{X: "NotEqual"},
		y:         ts.StructD{X: "not_equal"},
		opts:      []cmp.Option{addrTransform},
		wantEqual: true,
		reason:    "Equal method on StructD pointer called due to shallow copy transform",
	}, {
		label:     label + "/StructD/PointerEqual",
		x:         &ts.StructD{X: "NotEqual"},
		y:         &ts.StructD{X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructD pointer called",
	}, {
		label:     label + "/StructE/ValueInequal",
		x:         ts.StructE{X: "NotEqual"},
		y:         ts.StructE{X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructE value not called",
	}, {
		label:     label + "/StructE/ValueAddrEqual",
		x:         ts.StructE{X: "NotEqual"},
		y:         ts.StructE{X: "not_equal"},
		opts:      []cmp.Option{addrTransform},
		wantEqual: true,
		reason:    "Equal method on StructE pointer called due to shallow copy transform",
	}, {
		label:     label + "/StructE/PointerEqual",
		x:         &ts.StructE{X: "NotEqual"},
		y:         &ts.StructE{X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructE pointer called",
	}, {
		label:     label + "/StructF/ValueInequal",
		x:         ts.StructF{X: "NotEqual"},
		y:         ts.StructF{X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructF value not called",
	}, {
		label:     label + "/StructF/PointerEqual",
		x:         &ts.StructF{X: "NotEqual"},
		y:         &ts.StructF{X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructF pointer called",
	}, {
		label:     label + "/StructA1/ValueEqual",
		x:         ts.StructA1{StructA: ts.StructA{X: "NotEqual"}, X: "equal"},
		y:         ts.StructA1{StructA: ts.StructA{X: "not_equal"}, X: "equal"},
		wantEqual: true,
		reason:    "Equal method on StructA value called with equal X field",
	}, {
		label:     label + "/StructA1/ValueInequal",
		x:         ts.StructA1{StructA: ts.StructA{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructA1{StructA: ts.StructA{X: "not_equal"}, X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructA value called, but inequal X field",
	}, {
		label:     label + "/StructA1/PointerEqual",
		x:         &ts.StructA1{StructA: ts.StructA{X: "NotEqual"}, X: "equal"},
		y:         &ts.StructA1{StructA: ts.StructA{X: "not_equal"}, X: "equal"},
		wantEqual: true,
		reason:    "Equal method on StructA value called with equal X field",
	}, {
		label:     label + "/StructA1/PointerInequal",
		x:         &ts.StructA1{StructA: ts.StructA{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructA1{StructA: ts.StructA{X: "not_equal"}, X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructA value called, but inequal X field",
	}, {
		label:     label + "/StructB1/ValueEqual",
		x:         ts.StructB1{StructB: ts.StructB{X: "NotEqual"}, X: "equal"},
		y:         ts.StructB1{StructB: ts.StructB{X: "not_equal"}, X: "equal"},
		opts:      []cmp.Option{addrTransform},
		wantEqual: true,
		reason:    "Equal method on StructB pointer called due to shallow copy transform with equal X field",
	}, {
		label:     label + "/StructB1/ValueInequal",
		x:         ts.StructB1{StructB: ts.StructB{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructB1{StructB: ts.StructB{X: "not_equal"}, X: "not_equal"},
		opts:      []cmp.Option{addrTransform},
		wantEqual: false,
		reason:    "Equal method on StructB pointer called due to shallow copy transform, but inequal X field",
	}, {
		label:     label + "/StructB1/PointerEqual",
		x:         &ts.StructB1{StructB: ts.StructB{X: "NotEqual"}, X: "equal"},
		y:         &ts.StructB1{StructB: ts.StructB{X: "not_equal"}, X: "equal"},
		opts:      []cmp.Option{addrTransform},
		wantEqual: true,
		reason:    "Equal method on StructB pointer called due to shallow copy transform with equal X field",
	}, {
		label:     label + "/StructB1/PointerInequal",
		x:         &ts.StructB1{StructB: ts.StructB{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructB1{StructB: ts.StructB{X: "not_equal"}, X: "not_equal"},
		opts:      []cmp.Option{addrTransform},
		wantEqual: false,
		reason:    "Equal method on StructB pointer called due to shallow copy transform, but inequal X field",
	}, {
		label:     label + "/StructC1/ValueEqual",
		x:         ts.StructC1{StructC: ts.StructC{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructC1{StructC: ts.StructC{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructC1 value called",
	}, {
		label:     label + "/StructC1/PointerEqual",
		x:         &ts.StructC1{StructC: ts.StructC{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructC1{StructC: ts.StructC{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructC1 pointer called",
	}, {
		label:     label + "/StructD1/ValueInequal",
		x:         ts.StructD1{StructD: ts.StructD{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructD1{StructD: ts.StructD{X: "not_equal"}, X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructD1 value not called",
	}, {
		label:     label + "/StructD1/PointerAddrEqual",
		x:         ts.StructD1{StructD: ts.StructD{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructD1{StructD: ts.StructD{X: "not_equal"}, X: "not_equal"},
		opts:      []cmp.Option{addrTransform},
		wantEqual: true,
		reason:    "Equal method on StructD1 pointer called due to shallow copy transform",
	}, {
		label:     label + "/StructD1/PointerEqual",
		x:         &ts.StructD1{StructD: ts.StructD{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructD1{StructD: ts.StructD{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructD1 pointer called",
	}, {
		label:     label + "/StructE1/ValueInequal",
		x:         ts.StructE1{StructE: ts.StructE{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructE1{StructE: ts.StructE{X: "not_equal"}, X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructE1 value not called",
	}, {
		label:     label + "/StructE1/ValueAddrEqual",
		x:         ts.StructE1{StructE: ts.StructE{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructE1{StructE: ts.StructE{X: "not_equal"}, X: "not_equal"},
		opts:      []cmp.Option{addrTransform},
		wantEqual: true,
		reason:    "Equal method on StructE1 pointer called due to shallow copy transform",
	}, {
		label:     label + "/StructE1/PointerEqual",
		x:         &ts.StructE1{StructE: ts.StructE{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructE1{StructE: ts.StructE{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructE1 pointer called",
	}, {
		label:     label + "/StructF1/ValueInequal",
		x:         ts.StructF1{StructF: ts.StructF{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructF1{StructF: ts.StructF{X: "not_equal"}, X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructF1 value not called",
	}, {
		label:     label + "/StructF1/PointerEqual",
		x:         &ts.StructF1{StructF: ts.StructF{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructF1{StructF: ts.StructF{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method on StructF1 pointer called",
	}, {
		label:     label + "/StructA2/ValueEqual",
		x:         ts.StructA2{StructA: &ts.StructA{X: "NotEqual"}, X: "equal"},
		y:         ts.StructA2{StructA: &ts.StructA{X: "not_equal"}, X: "equal"},
		wantEqual: true,
		reason:    "Equal method on StructA pointer called with equal X field",
	}, {
		label:     label + "/StructA2/ValueInequal",
		x:         ts.StructA2{StructA: &ts.StructA{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructA2{StructA: &ts.StructA{X: "not_equal"}, X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructA pointer called, but inequal X field",
	}, {
		label:     label + "/StructA2/PointerEqual",
		x:         &ts.StructA2{StructA: &ts.StructA{X: "NotEqual"}, X: "equal"},
		y:         &ts.StructA2{StructA: &ts.StructA{X: "not_equal"}, X: "equal"},
		wantEqual: true,
		reason:    "Equal method on StructA pointer called with equal X field",
	}, {
		label:     label + "/StructA2/PointerInequal",
		x:         &ts.StructA2{StructA: &ts.StructA{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructA2{StructA: &ts.StructA{X: "not_equal"}, X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructA pointer called, but inequal X field",
	}, {
		label:     label + "/StructB2/ValueEqual",
		x:         ts.StructB2{StructB: &ts.StructB{X: "NotEqual"}, X: "equal"},
		y:         ts.StructB2{StructB: &ts.StructB{X: "not_equal"}, X: "equal"},
		wantEqual: true,
		reason:    "Equal method on StructB pointer called with equal X field",
	}, {
		label:     label + "/StructB2/ValueInequal",
		x:         ts.StructB2{StructB: &ts.StructB{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructB2{StructB: &ts.StructB{X: "not_equal"}, X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructB pointer called, but inequal X field",
	}, {
		label:     label + "/StructB2/PointerEqual",
		x:         &ts.StructB2{StructB: &ts.StructB{X: "NotEqual"}, X: "equal"},
		y:         &ts.StructB2{StructB: &ts.StructB{X: "not_equal"}, X: "equal"},
		wantEqual: true,
		reason:    "Equal method on StructB pointer called with equal X field",
	}, {
		label:     label + "/StructB2/PointerInequal",
		x:         &ts.StructB2{StructB: &ts.StructB{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructB2{StructB: &ts.StructB{X: "not_equal"}, X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method on StructB pointer called, but inequal X field",
	}, {
		label:     label + "/StructC2/ValueEqual",
		x:         ts.StructC2{StructC: &ts.StructC{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructC2{StructC: &ts.StructC{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method called on StructC2 value due to forwarded StructC pointer",
	}, {
		label:     label + "/StructC2/PointerEqual",
		x:         &ts.StructC2{StructC: &ts.StructC{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructC2{StructC: &ts.StructC{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method called on StructC2 pointer due to forwarded StructC pointer",
	}, {
		label:     label + "/StructD2/ValueEqual",
		x:         ts.StructD2{StructD: &ts.StructD{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructD2{StructD: &ts.StructD{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method called on StructD2 value due to forwarded StructD pointer",
	}, {
		label:     label + "/StructD2/PointerEqual",
		x:         &ts.StructD2{StructD: &ts.StructD{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructD2{StructD: &ts.StructD{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method called on StructD2 pointer due to forwarded StructD pointer",
	}, {
		label:     label + "/StructE2/ValueEqual",
		x:         ts.StructE2{StructE: &ts.StructE{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructE2{StructE: &ts.StructE{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method called on StructE2 value due to forwarded StructE pointer",
	}, {
		label:     label + "/StructE2/PointerEqual",
		x:         &ts.StructE2{StructE: &ts.StructE{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructE2{StructE: &ts.StructE{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method called on StructE2 pointer due to forwarded StructE pointer",
	}, {
		label:     label + "/StructF2/ValueEqual",
		x:         ts.StructF2{StructF: &ts.StructF{X: "NotEqual"}, X: "NotEqual"},
		y:         ts.StructF2{StructF: &ts.StructF{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method called on StructF2 value due to forwarded StructF pointer",
	}, {
		label:     label + "/StructF2/PointerEqual",
		x:         &ts.StructF2{StructF: &ts.StructF{X: "NotEqual"}, X: "NotEqual"},
		y:         &ts.StructF2{StructF: &ts.StructF{X: "not_equal"}, X: "not_equal"},
		wantEqual: true,
		reason:    "Equal method called on StructF2 pointer due to forwarded StructF pointer",
	}, {
		label:     label + "/StructNo/Inequal",
		x:         ts.StructNo{X: "NotEqual"},
		y:         ts.StructNo{X: "not_equal"},
		wantEqual: false,
		reason:    "Equal method not called since StructNo is not assignable to InterfaceA",
	}, {
		label:     label + "/AssignA/Equal",
		x:         ts.AssignA(func() int { return 0 }),
		y:         ts.AssignA(func() int { return 1 }),
		wantEqual: true,
		reason:    "Equal method called since named func is assignable to unnamed func",
	}, {
		label:     label + "/AssignB/Equal",
		x:         ts.AssignB(struct{ A int }{0}),
		y:         ts.AssignB(struct{ A int }{1}),
		wantEqual: true,
		reason:    "Equal method called since named struct is assignable to unnamed struct",
	}, {
		label:     label + "/AssignC/Equal",
		x:         ts.AssignC(make(chan bool)),
		y:         ts.AssignC(make(chan bool)),
		wantEqual: true,
		reason:    "Equal method called since named channel is assignable to unnamed channel",
	}, {
		label:     label + "/AssignD/Equal",
		x:         ts.AssignD(make(chan bool)),
		y:         ts.AssignD(make(chan bool)),
		wantEqual: true,
		reason:    "Equal method called since named channel is assignable to unnamed channel",
	}}
}

type (
	CycleAlpha struct {
		Name   string
		Bravos map[string]*CycleBravo
	}
	CycleBravo struct {
		ID     int
		Name   string
		Mods   int
		Alphas map[string]*CycleAlpha
	}
)

func cycleTests() []test {
	const label = "Cycle"

	type (
		P *P
		S []S
		M map[int]M
	)

	makeGraph := func() map[string]*CycleAlpha {
		v := map[string]*CycleAlpha{
			"Foo": &CycleAlpha{
				Name: "Foo",
				Bravos: map[string]*CycleBravo{
					"FooBravo": &CycleBravo{
						Name: "FooBravo",
						ID:   101,
						Mods: 100,
						Alphas: map[string]*CycleAlpha{
							"Foo": nil, // cyclic reference
						},
					},
				},
			},
			"Bar": &CycleAlpha{
				Name: "Bar",
				Bravos: map[string]*CycleBravo{
					"BarBuzzBravo": &CycleBravo{
						Name: "BarBuzzBravo",
						ID:   102,
						Mods: 2,
						Alphas: map[string]*CycleAlpha{
							"Bar":  nil, // cyclic reference
							"Buzz": nil, // cyclic reference
						},
					},
					"BuzzBarBravo": &CycleBravo{
						Name: "BuzzBarBravo",
						ID:   103,
						Mods: 0,
						Alphas: map[string]*CycleAlpha{
							"Bar":  nil, // cyclic reference
							"Buzz": nil, // cyclic reference
						},
					},
				},
			},
			"Buzz": &CycleAlpha{
				Name: "Buzz",
				Bravos: map[string]*CycleBravo{
					"BarBuzzBravo": nil, // cyclic reference
					"BuzzBarBravo": nil, // cyclic reference
				},
			},
		}
		v["Foo"].Bravos["FooBravo"].Alphas["Foo"] = v["Foo"]
		v["Bar"].Bravos["BarBuzzBravo"].Alphas["Bar"] = v["Bar"]
		v["Bar"].Bravos["BarBuzzBravo"].Alphas["Buzz"] = v["Buzz"]
		v["Bar"].Bravos["BuzzBarBravo"].Alphas["Bar"] = v["Bar"]
		v["Bar"].Bravos["BuzzBarBravo"].Alphas["Buzz"] = v["Buzz"]
		v["Buzz"].Bravos["BarBuzzBravo"] = v["Bar"].Bravos["BarBuzzBravo"]
		v["Buzz"].Bravos["BuzzBarBravo"] = v["Bar"].Bravos["BuzzBarBravo"]
		return v
	}

	var tests []test
	type XY struct{ x, y interface{} }
	for _, tt := range []struct {
		label     string
		in        XY
		wantEqual bool
		reason    string
	}{{
		label: "PointersEqual",
		in: func() XY {
			x := new(P)
			*x = x
			y := new(P)
			*y = y
			return XY{x, y}
		}(),
		wantEqual: true,
		reason:    "equal pair of single-node pointers",
	}, {
		label: "PointersInequal",
		in: func() XY {
			x := new(P)
			*x = x
			y1, y2 := new(P), new(P)
			*y1 = y2
			*y2 = y1
			return XY{x, y1}
		}(),
		wantEqual: false,
		reason:    "inequal pair of single-node and double-node pointers",
	}, {
		label: "SlicesEqual",
		in: func() XY {
			x := S{nil}
			x[0] = x
			y := S{nil}
			y[0] = y
			return XY{x, y}
		}(),
		wantEqual: true,
		reason:    "equal pair of single-node slices",
	}, {
		label: "SlicesInequal",
		in: func() XY {
			x := S{nil}
			x[0] = x
			y1, y2 := S{nil}, S{nil}
			y1[0] = y2
			y2[0] = y1
			return XY{x, y1}
		}(),
		wantEqual: false,
		reason:    "inequal pair of single-node and double node slices",
	}, {
		label: "MapsEqual",
		in: func() XY {
			x := M{0: nil}
			x[0] = x
			y := M{0: nil}
			y[0] = y
			return XY{x, y}
		}(),
		wantEqual: true,
		reason:    "equal pair of single-node maps",
	}, {
		label: "MapsInequal",
		in: func() XY {
			x := M{0: nil}
			x[0] = x
			y1, y2 := M{0: nil}, M{0: nil}
			y1[0] = y2
			y2[0] = y1
			return XY{x, y1}
		}(),
		wantEqual: false,
		reason:    "inequal pair of single-node and double-node maps",
	}, {
		label:     "GraphEqual",
		in:        XY{makeGraph(), makeGraph()},
		wantEqual: true,
		reason:    "graphs are equal since they have identical forms",
	}, {
		label: "GraphInequalZeroed",
		in: func() XY {
			x := makeGraph()
			y := makeGraph()
			y["Foo"].Bravos["FooBravo"].ID = 0
			y["Bar"].Bravos["BarBuzzBravo"].ID = 0
			y["Bar"].Bravos["BuzzBarBravo"].ID = 0
			return XY{x, y}
		}(),
		wantEqual: false,
		reason:    "graphs are inequal because the ID fields are different",
	}, {
		label: "GraphInequalStruct",
		in: func() XY {
			x := makeGraph()
			y := makeGraph()
			x["Buzz"].Bravos["BuzzBarBravo"] = &CycleBravo{
				Name: "BuzzBarBravo",
				ID:   103,
			}
			return XY{x, y}
		}(),
		wantEqual: false,
		reason:    "graphs are inequal because they differ on a map element",
	}} {
		tests = append(tests, test{
			label:     label + "/" + tt.label,
			x:         tt.in.x,
			y:         tt.in.y,
			wantEqual: tt.wantEqual,
			reason:    tt.reason,
		})
	}
	return tests
}

func project1Tests() []test {
	const label = "Project1"

	ignoreUnexported := cmpopts.IgnoreUnexported(
		ts.EagleImmutable{},
		ts.DreamerImmutable{},
		ts.SlapImmutable{},
		ts.GoatImmutable{},
		ts.DonkeyImmutable{},
		ts.LoveRadius{},
		ts.SummerLove{},
		ts.SummerLoveSummary{},
	)

	createEagle := func() ts.Eagle {
		return ts.Eagle{
			Name:   "eagle",
			Hounds: []string{"buford", "tannen"},
			Desc:   "some description",
			Dreamers: []ts.Dreamer{{}, {
				Name: "dreamer2",
				Animal: []interface{}{
					ts.Goat{
						Target: "corporation",
						Immutable: &ts.GoatImmutable{
							ID:      "southbay",
							State:   (*pb.Goat_States)(newInt(5)),
							Started: now,
						},
					},
					ts.Donkey{},
				},
				Amoeba: 53,
			}},
			Slaps: []ts.Slap{{
				Name: "slapID",
				Args: &pb.MetaData{Stringer: pb.Stringer{X: "metadata"}},
				Immutable: &ts.SlapImmutable{
					ID:       "immutableSlap",
					MildSlap: true,
					Started:  now,
					LoveRadius: &ts.LoveRadius{
						Summer: &ts.SummerLove{
							Summary: &ts.SummerLoveSummary{
								Devices:    []string{"foo", "bar", "baz"},
								ChangeType: []pb.SummerType{1, 2, 3},
							},
						},
					},
				},
			}},
			Immutable: &ts.EagleImmutable{
				ID:          "eagleID",
				Birthday:    now,
				MissingCall: (*pb.Eagle_MissingCalls)(newInt(55)),
			},
		}
	}

	return []test{{
		label: label + "/PanicUnexported",
		x: ts.Eagle{Slaps: []ts.Slap{{
			Args: &pb.MetaData{Stringer: pb.Stringer{X: "metadata"}},
		}}},
		y: ts.Eagle{Slaps: []ts.Slap{{
			Args: &pb.MetaData{Stringer: pb.Stringer{X: "metadata"}},
		}}},
		wantPanic: "cannot handle unexported field",
		reason:    "struct contains unexported fields",
	}, {
		label: label + "/ProtoEqual",
		x: ts.Eagle{Slaps: []ts.Slap{{
			Args: &pb.MetaData{Stringer: pb.Stringer{X: "metadata"}},
		}}},
		y: ts.Eagle{Slaps: []ts.Slap{{
			Args: &pb.MetaData{Stringer: pb.Stringer{X: "metadata"}},
		}}},
		opts:      []cmp.Option{cmp.Comparer(pb.Equal)},
		wantEqual: true,
		reason:    "simulated protobuf messages contain the same values",
	}, {
		label: label + "/ProtoInequal",
		x: ts.Eagle{Slaps: []ts.Slap{{}, {}, {}, {}, {
			Args: &pb.MetaData{Stringer: pb.Stringer{X: "metadata"}},
		}}},
		y: ts.Eagle{Slaps: []ts.Slap{{}, {}, {}, {}, {
			Args: &pb.MetaData{Stringer: pb.Stringer{X: "metadata2"}},
		}}},
		opts:      []cmp.Option{cmp.Comparer(pb.Equal)},
		wantEqual: false,
		reason:    "simulated protobuf messages contain different values",
	}, {
		label:     label + "/Equal",
		x:         createEagle(),
		y:         createEagle(),
		opts:      []cmp.Option{ignoreUnexported, cmp.Comparer(pb.Equal)},
		wantEqual: true,
		reason:    "equal because values are the same",
	}, {
		label: label + "/Inequal",
		x: func() ts.Eagle {
			eg := createEagle()
			eg.Dreamers[1].Animal[0].(ts.Goat).Immutable.ID = "southbay2"
			eg.Dreamers[1].Animal[0].(ts.Goat).Immutable.State = (*pb.Goat_States)(newInt(6))
			eg.Slaps[0].Immutable.MildSlap = false
			return eg
		}(),
		y: func() ts.Eagle {
			eg := createEagle()
			devs := eg.Slaps[0].Immutable.LoveRadius.Summer.Summary.Devices
			eg.Slaps[0].Immutable.LoveRadius.Summer.Summary.Devices = devs[:1]
			return eg
		}(),
		opts:      []cmp.Option{ignoreUnexported, cmp.Comparer(pb.Equal)},
		wantEqual: false,
		reason:    "inequal because some values are different",
	}}
}

type germSorter []*pb.Germ

func (gs germSorter) Len() int           { return len(gs) }
func (gs germSorter) Less(i, j int) bool { return gs[i].String() < gs[j].String() }
func (gs germSorter) Swap(i, j int)      { gs[i], gs[j] = gs[j], gs[i] }

func project2Tests() []test {
	const label = "Project2"

	sortGerms := cmp.Transformer("Sort", func(in []*pb.Germ) []*pb.Germ {
		out := append([]*pb.Germ(nil), in...) // Make copy
		sort.Sort(germSorter(out))
		return out
	})

	equalDish := cmp.Comparer(func(x, y *ts.Dish) bool {
		if x == nil || y == nil {
			return x == nil && y == nil
		}
		px, err1 := x.Proto()
		py, err2 := y.Proto()
		if err1 != nil || err2 != nil {
			return err1 == err2
		}
		return pb.Equal(px, py)
	})

	createBatch := func() ts.GermBatch {
		return ts.GermBatch{
			DirtyGerms: map[int32][]*pb.Germ{
				17: {
					{Stringer: pb.Stringer{X: "germ1"}},
				},
				18: {
					{Stringer: pb.Stringer{X: "germ2"}},
					{Stringer: pb.Stringer{X: "germ3"}},
					{Stringer: pb.Stringer{X: "germ4"}},
				},
			},
			GermMap: map[int32]*pb.Germ{
				13: {Stringer: pb.Stringer{X: "germ13"}},
				21: {Stringer: pb.Stringer{X: "germ21"}},
			},
			DishMap: map[int32]*ts.Dish{
				0: ts.CreateDish(nil, io.EOF),
				1: ts.CreateDish(nil, io.ErrUnexpectedEOF),
				2: ts.CreateDish(&pb.Dish{Stringer: pb.Stringer{X: "dish"}}, nil),
			},
			HasPreviousResult: true,
			DirtyID:           10,
			GermStrain:        421,
			InfectedAt:        now,
		}
	}

	return []test{{
		label:     label + "/PanicUnexported",
		x:         createBatch(),
		y:         createBatch(),
		wantPanic: "cannot handle unexported field",
		reason:    "struct contains unexported fields",
	}, {
		label:     label + "/Equal",
		x:         createBatch(),
		y:         createBatch(),
		opts:      []cmp.Option{cmp.Comparer(pb.Equal), sortGerms, equalDish},
		wantEqual: true,
		reason:    "equal because identical values are compared",
	}, {
		label: label + "/InequalOrder",
		x:     createBatch(),
		y: func() ts.GermBatch {
			gb := createBatch()
			s := gb.DirtyGerms[18]
			s[0], s[1], s[2] = s[1], s[2], s[0]
			return gb
		}(),
		opts:      []cmp.Option{cmp.Comparer(pb.Equal), equalDish},
		wantEqual: false,
		reason:    "inequal because slice contains elements in differing order",
	}, {
		label: label + "/EqualOrder",
		x:     createBatch(),
		y: func() ts.GermBatch {
			gb := createBatch()
			s := gb.DirtyGerms[18]
			s[0], s[1], s[2] = s[1], s[2], s[0]
			return gb
		}(),
		opts:      []cmp.Option{cmp.Comparer(pb.Equal), sortGerms, equalDish},
		wantEqual: true,
		reason:    "equal because unordered slice is sorted using transformer",
	}, {
		label: label + "/Inequal",
		x: func() ts.GermBatch {
			gb := createBatch()
			delete(gb.DirtyGerms, 17)
			gb.DishMap[1] = nil
			return gb
		}(),
		y: func() ts.GermBatch {
			gb := createBatch()
			gb.DirtyGerms[18] = gb.DirtyGerms[18][:2]
			gb.GermStrain = 22
			return gb
		}(),
		opts:      []cmp.Option{cmp.Comparer(pb.Equal), sortGerms, equalDish},
		wantEqual: false,
		reason:    "inequal because some values are different",
	}}
}

func project3Tests() []test {
	const label = "Project3"

	allowVisibility := cmp.AllowUnexported(ts.Dirt{})

	ignoreLocker := cmpopts.IgnoreInterfaces(struct{ sync.Locker }{})

	transformProtos := cmp.Transformer("λ", func(x pb.Dirt) *pb.Dirt {
		return &x
	})

	equalTable := cmp.Comparer(func(x, y ts.Table) bool {
		tx, ok1 := x.(*ts.MockTable)
		ty, ok2 := y.(*ts.MockTable)
		if !ok1 || !ok2 {
			panic("table type must be MockTable")
		}
		return cmp.Equal(tx.State(), ty.State())
	})

	createDirt := func() (d ts.Dirt) {
		d.SetTable(ts.CreateMockTable([]string{"a", "b", "c"}))
		d.SetTimestamp(12345)
		d.Discord = 554
		d.Proto = pb.Dirt{Stringer: pb.Stringer{X: "proto"}}
		d.SetWizard(map[string]*pb.Wizard{
			"harry": {Stringer: pb.Stringer{X: "potter"}},
			"albus": {Stringer: pb.Stringer{X: "dumbledore"}},
		})
		d.SetLastTime(54321)
		return d
	}

	return []test{{
		label:     label + "/PanicUnexported1",
		x:         createDirt(),
		y:         createDirt(),
		wantPanic: "cannot handle unexported field",
		reason:    "struct contains unexported fields",
	}, {
		label:     label + "/PanicUnexported2",
		x:         createDirt(),
		y:         createDirt(),
		opts:      []cmp.Option{allowVisibility, ignoreLocker, cmp.Comparer(pb.Equal), equalTable},
		wantPanic: "cannot handle unexported field",
		reason:    "struct contains references to simulated protobuf types with unexported fields",
	}, {
		label:     label + "/Equal",
		x:         createDirt(),
		y:         createDirt(),
		opts:      []cmp.Option{allowVisibility, transformProtos, ignoreLocker, cmp.Comparer(pb.Equal), equalTable},
		wantEqual: true,
		reason:    "transformer used to create reference to protobuf message so it works with pb.Equal",
	}, {
		label: label + "/Inequal",
		x: func() ts.Dirt {
			d := createDirt()
			d.SetTable(ts.CreateMockTable([]string{"a", "c"}))
			d.Proto = pb.Dirt{Stringer: pb.Stringer{X: "blah"}}
			return d
		}(),
		y: func() ts.Dirt {
			d := createDirt()
			d.Discord = 500
			d.SetWizard(map[string]*pb.Wizard{
				"harry": {Stringer: pb.Stringer{X: "otter"}},
			})
			return d
		}(),
		opts:      []cmp.Option{allowVisibility, transformProtos, ignoreLocker, cmp.Comparer(pb.Equal), equalTable},
		wantEqual: false,
		reason:    "inequal because some values are different",
	}}
}

func project4Tests() []test {
	const label = "Project4"

	allowVisibility := cmp.AllowUnexported(
		ts.Cartel{},
		ts.Headquarter{},
		ts.Poison{},
	)

	transformProtos := cmp.Transformer("λ", func(x pb.Restrictions) *pb.Restrictions {
		return &x
	})

	createCartel := func() ts.Cartel {
		var p ts.Poison
		p.SetPoisonType(5)
		p.SetExpiration(now)
		p.SetManufacturer("acme")

		var hq ts.Headquarter
		hq.SetID(5)
		hq.SetLocation("moon")
		hq.SetSubDivisions([]string{"alpha", "bravo", "charlie"})
		hq.SetMetaData(&pb.MetaData{Stringer: pb.Stringer{X: "metadata"}})
		hq.SetPublicMessage([]byte{1, 2, 3, 4, 5})
		hq.SetHorseBack("abcdef")
		hq.SetStatus(44)

		var c ts.Cartel
		c.Headquarter = hq
		c.SetSource("mars")
		c.SetCreationTime(now)
		c.SetBoss("al capone")
		c.SetPoisons([]*ts.Poison{&p})

		return c
	}

	return []test{{
		label:     label + "/PanicUnexported1",
		x:         createCartel(),
		y:         createCartel(),
		wantPanic: "cannot handle unexported field",
		reason:    "struct contains unexported fields",
	}, {
		label:     label + "/PanicUnexported2",
		x:         createCartel(),
		y:         createCartel(),
		opts:      []cmp.Option{allowVisibility, cmp.Comparer(pb.Equal)},
		wantPanic: "cannot handle unexported field",
		reason:    "struct contains references to simulated protobuf types with unexported fields",
	}, {
		label:     label + "/Equal",
		x:         createCartel(),
		y:         createCartel(),
		opts:      []cmp.Option{allowVisibility, transformProtos, cmp.Comparer(pb.Equal)},
		wantEqual: true,
		reason:    "transformer used to create reference to protobuf message so it works with pb.Equal",
	}, {
		label: label + "/Inequal",
		x: func() ts.Cartel {
			d := createCartel()
			var p1, p2 ts.Poison
			p1.SetPoisonType(1)
			p1.SetExpiration(now)
			p1.SetManufacturer("acme")
			p2.SetPoisonType(2)
			p2.SetManufacturer("acme2")
			d.SetPoisons([]*ts.Poison{&p1, &p2})
			return d
		}(),
		y: func() ts.Cartel {
			d := createCartel()
			d.SetSubDivisions([]string{"bravo", "charlie"})
			d.SetPublicMessage([]byte{1, 2, 4, 3, 5})
			return d
		}(),
		opts:      []cmp.Option{allowVisibility, transformProtos, cmp.Comparer(pb.Equal)},
		wantEqual: false,
		reason:    "inequal because some values are different",
	}}
}

// BenchmarkBytes benchmarks the performance of performing Equal or Diff on
// large slices of bytes.
func BenchmarkBytes(b *testing.B) {
	// Create a list of PathFilters that never apply, but are evaluated.
	const maxFilters = 5
	var filters cmp.Options
	errorIface := reflect.TypeOf((*error)(nil)).Elem()
	for i := 0; i <= maxFilters; i++ {
		filters = append(filters, cmp.FilterPath(func(p cmp.Path) bool {
			return p.Last().Type().AssignableTo(errorIface) // Never true
		}, cmp.Ignore()))
	}

	type benchSize struct {
		label string
		size  int64
	}
	for _, ts := range []benchSize{
		{"4KiB", 1 << 12},
		{"64KiB", 1 << 16},
		{"1MiB", 1 << 20},
		{"16MiB", 1 << 24},
	} {
		bx := append(append(make([]byte, ts.size/2), 'x'), make([]byte, ts.size/2)...)
		by := append(append(make([]byte, ts.size/2), 'y'), make([]byte, ts.size/2)...)
		b.Run(ts.label, func(b *testing.B) {
			// Iteratively add more filters that never apply, but are evaluated
			// to measure the cost of simply evaluating each filter.
			for i := 0; i <= maxFilters; i++ {
				b.Run(fmt.Sprintf("EqualFilter%d", i), func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(2 * ts.size)
					for j := 0; j < b.N; j++ {
						cmp.Equal(bx, by, filters[:i]...)
					}
				})
			}
			for i := 0; i <= maxFilters; i++ {
				b.Run(fmt.Sprintf("DiffFilter%d", i), func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(2 * ts.size)
					for j := 0; j < b.N; j++ {
						cmp.Diff(bx, by, filters[:i]...)
					}
				})
			}
		})
	}
}
