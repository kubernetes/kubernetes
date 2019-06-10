package semver

import (
	"reflect"
	"strings"
	"testing"
)

type wildcardTypeTest struct {
	input        string
	wildcardType wildcardType
}

type comparatorTest struct {
	input      string
	comparator func(comparator) bool
}

func TestParseComparator(t *testing.T) {
	compatorTests := []comparatorTest{
		{">", testGT},
		{">=", testGE},
		{"<", testLT},
		{"<=", testLE},
		{"", testEQ},
		{"=", testEQ},
		{"==", testEQ},
		{"!=", testNE},
		{"!", testNE},
		{"-", nil},
		{"<==", nil},
		{"<<", nil},
		{">>", nil},
	}

	for _, tc := range compatorTests {
		if c := parseComparator(tc.input); c == nil {
			if tc.comparator != nil {
				t.Errorf("Comparator nil for case %q\n", tc.input)
			}
		} else if !tc.comparator(c) {
			t.Errorf("Invalid comparator for case %q\n", tc.input)
		}
	}
}

var (
	v1 = MustParse("1.2.2")
	v2 = MustParse("1.2.3")
	v3 = MustParse("1.2.4")
)

func testEQ(f comparator) bool {
	return f(v1, v1) && !f(v1, v2)
}

func testNE(f comparator) bool {
	return !f(v1, v1) && f(v1, v2)
}

func testGT(f comparator) bool {
	return f(v2, v1) && f(v3, v2) && !f(v1, v2) && !f(v1, v1)
}

func testGE(f comparator) bool {
	return f(v2, v1) && f(v3, v2) && !f(v1, v2)
}

func testLT(f comparator) bool {
	return f(v1, v2) && f(v2, v3) && !f(v2, v1) && !f(v1, v1)
}

func testLE(f comparator) bool {
	return f(v1, v2) && f(v2, v3) && !f(v2, v1)
}

func TestSplitAndTrim(t *testing.T) {
	tests := []struct {
		i string
		s []string
	}{
		{"1.2.3 1.2.3", []string{"1.2.3", "1.2.3"}},
		{"     1.2.3     1.2.3     ", []string{"1.2.3", "1.2.3"}},       // Spaces
		{"  >=   1.2.3   <=  1.2.3   ", []string{">=1.2.3", "<=1.2.3"}}, // Spaces between operator and version
		{"1.2.3 || >=1.2.3 <1.2.3", []string{"1.2.3", "||", ">=1.2.3", "<1.2.3"}},
		{"      1.2.3      ||     >=1.2.3     <1.2.3    ", []string{"1.2.3", "||", ">=1.2.3", "<1.2.3"}},
	}

	for _, tc := range tests {
		p := splitAndTrim(tc.i)
		if !reflect.DeepEqual(p, tc.s) {
			t.Errorf("Invalid for case %q: Expected %q, got: %q", tc.i, tc.s, p)
		}
	}
}

func TestSplitComparatorVersion(t *testing.T) {
	tests := []struct {
		i string
		p []string
	}{
		{">1.2.3", []string{">", "1.2.3"}},
		{">=1.2.3", []string{">=", "1.2.3"}},
		{"<1.2.3", []string{"<", "1.2.3"}},
		{"<=1.2.3", []string{"<=", "1.2.3"}},
		{"1.2.3", []string{"", "1.2.3"}},
		{"=1.2.3", []string{"=", "1.2.3"}},
		{"==1.2.3", []string{"==", "1.2.3"}},
		{"!=1.2.3", []string{"!=", "1.2.3"}},
		{"!1.2.3", []string{"!", "1.2.3"}},
		{"error", nil},
	}
	for _, tc := range tests {
		if op, v, err := splitComparatorVersion(tc.i); err != nil {
			if tc.p != nil {
				t.Errorf("Invalid for case %q: Expected %q, got error %q", tc.i, tc.p, err)
			}
		} else if op != tc.p[0] {
			t.Errorf("Invalid operator for case %q: Expected %q, got: %q", tc.i, tc.p[0], op)
		} else if v != tc.p[1] {
			t.Errorf("Invalid version for case %q: Expected %q, got: %q", tc.i, tc.p[1], v)
		}

	}
}

func TestBuildVersionRange(t *testing.T) {
	tests := []struct {
		opStr string
		vStr  string
		c     func(comparator) bool
		v     string
	}{
		{">", "1.2.3", testGT, "1.2.3"},
		{">=", "1.2.3", testGE, "1.2.3"},
		{"<", "1.2.3", testLT, "1.2.3"},
		{"<=", "1.2.3", testLE, "1.2.3"},
		{"", "1.2.3", testEQ, "1.2.3"},
		{"=", "1.2.3", testEQ, "1.2.3"},
		{"==", "1.2.3", testEQ, "1.2.3"},
		{"!=", "1.2.3", testNE, "1.2.3"},
		{"!", "1.2.3", testNE, "1.2.3"},
		{">>", "1.2.3", nil, ""},  // Invalid comparator
		{"=", "invalid", nil, ""}, // Invalid version
	}

	for _, tc := range tests {
		if r, err := buildVersionRange(tc.opStr, tc.vStr); err != nil {
			if tc.c != nil {
				t.Errorf("Invalid for case %q: Expected %q, got error %q", strings.Join([]string{tc.opStr, tc.vStr}, ""), tc.v, err)
			}
		} else if r == nil {
			t.Errorf("Invalid for case %q: got nil", strings.Join([]string{tc.opStr, tc.vStr}, ""))
		} else {
			// test version
			if tv := MustParse(tc.v); !r.v.EQ(tv) {
				t.Errorf("Invalid for case %q: Expected version %q, got: %q", strings.Join([]string{tc.opStr, tc.vStr}, ""), tv, r.v)
			}
			// test comparator
			if r.c == nil {
				t.Errorf("Invalid for case %q: got nil comparator", strings.Join([]string{tc.opStr, tc.vStr}, ""))
				continue
			}
			if !tc.c(r.c) {
				t.Errorf("Invalid comparator for case %q\n", strings.Join([]string{tc.opStr, tc.vStr}, ""))
			}
		}
	}

}

func TestSplitORParts(t *testing.T) {
	tests := []struct {
		i []string
		o [][]string
	}{
		{[]string{">1.2.3", "||", "<1.2.3", "||", "=1.2.3"}, [][]string{
			[]string{">1.2.3"},
			[]string{"<1.2.3"},
			[]string{"=1.2.3"},
		}},
		{[]string{">1.2.3", "<1.2.3", "||", "=1.2.3"}, [][]string{
			[]string{">1.2.3", "<1.2.3"},
			[]string{"=1.2.3"},
		}},
		{[]string{">1.2.3", "||"}, nil},
		{[]string{"||", ">1.2.3"}, nil},
	}
	for _, tc := range tests {
		o, err := splitORParts(tc.i)
		if err != nil && tc.o != nil {
			t.Errorf("Unexpected error for case %q: %s", tc.i, err)
		}
		if !reflect.DeepEqual(tc.o, o) {
			t.Errorf("Invalid for case %q: Expected %q, got: %q", tc.i, tc.o, o)
		}
	}
}

func TestGetWildcardType(t *testing.T) {
	wildcardTypeTests := []wildcardTypeTest{
		{"x", majorWildcard},
		{"1.x", minorWildcard},
		{"1.2.x", patchWildcard},
		{"fo.o.b.ar", noneWildcard},
	}

	for _, tc := range wildcardTypeTests {
		o := getWildcardType(tc.input)
		if o != tc.wildcardType {
			t.Errorf("Invalid for case: %q: Expected %q, got: %q", tc.input, tc.wildcardType, o)
		}
	}
}

func TestCreateVersionFromWildcard(t *testing.T) {
	tests := []struct {
		i string
		s string
	}{
		{"1.2.x", "1.2.0"},
		{"1.x", "1.0.0"},
	}

	for _, tc := range tests {
		p := createVersionFromWildcard(tc.i)
		if p != tc.s {
			t.Errorf("Invalid for case %q: Expected %q, got: %q", tc.i, tc.s, p)
		}
	}
}

func TestIncrementMajorVersion(t *testing.T) {
	tests := []struct {
		i string
		s string
	}{
		{"1.2.3", "2.2.3"},
		{"1.2", "2.2"},
		{"foo.bar", ""},
	}

	for _, tc := range tests {
		p, _ := incrementMajorVersion(tc.i)
		if p != tc.s {
			t.Errorf("Invalid for case %q: Expected %q, got: %q", tc.i, tc.s, p)
		}
	}
}

func TestIncrementMinorVersion(t *testing.T) {
	tests := []struct {
		i string
		s string
	}{
		{"1.2.3", "1.3.3"},
		{"1.2", "1.3"},
		{"foo.bar", ""},
	}

	for _, tc := range tests {
		p, _ := incrementMinorVersion(tc.i)
		if p != tc.s {
			t.Errorf("Invalid for case %q: Expected %q, got: %q", tc.i, tc.s, p)
		}
	}
}

func TestExpandWildcardVersion(t *testing.T) {
	tests := []struct {
		i [][]string
		o [][]string
	}{
		{[][]string{[]string{"foox"}}, nil},
		{[][]string{[]string{">=1.2.x"}}, [][]string{[]string{">=1.2.0"}}},
		{[][]string{[]string{"<=1.2.x"}}, [][]string{[]string{"<1.3.0"}}},
		{[][]string{[]string{">1.2.x"}}, [][]string{[]string{">=1.3.0"}}},
		{[][]string{[]string{"<1.2.x"}}, [][]string{[]string{"<1.2.0"}}},
		{[][]string{[]string{"!=1.2.x"}}, [][]string{[]string{"<1.2.0", ">=1.3.0"}}},
		{[][]string{[]string{">=1.x"}}, [][]string{[]string{">=1.0.0"}}},
		{[][]string{[]string{"<=1.x"}}, [][]string{[]string{"<2.0.0"}}},
		{[][]string{[]string{">1.x"}}, [][]string{[]string{">=2.0.0"}}},
		{[][]string{[]string{"<1.x"}}, [][]string{[]string{"<1.0.0"}}},
		{[][]string{[]string{"!=1.x"}}, [][]string{[]string{"<1.0.0", ">=2.0.0"}}},
		{[][]string{[]string{"1.2.x"}}, [][]string{[]string{">=1.2.0", "<1.3.0"}}},
		{[][]string{[]string{"1.x"}}, [][]string{[]string{">=1.0.0", "<2.0.0"}}},
	}

	for _, tc := range tests {
		o, _ := expandWildcardVersion(tc.i)
		if !reflect.DeepEqual(tc.o, o) {
			t.Errorf("Invalid for case %q: Expected %q, got: %q", tc.i, tc.o, o)
		}
	}
}

func TestVersionRangeToRange(t *testing.T) {
	vr := versionRange{
		v: MustParse("1.2.3"),
		c: compLT,
	}
	rf := vr.rangeFunc()
	if !rf(MustParse("1.2.2")) || rf(MustParse("1.2.3")) {
		t.Errorf("Invalid conversion to range func")
	}
}

func TestRangeAND(t *testing.T) {
	v := MustParse("1.2.2")
	v1 := MustParse("1.2.1")
	v2 := MustParse("1.2.3")
	rf1 := Range(func(v Version) bool {
		return v.GT(v1)
	})
	rf2 := Range(func(v Version) bool {
		return v.LT(v2)
	})
	rf := rf1.AND(rf2)
	if rf(v1) {
		t.Errorf("Invalid rangefunc, accepted: %s", v1)
	}
	if rf(v2) {
		t.Errorf("Invalid rangefunc, accepted: %s", v2)
	}
	if !rf(v) {
		t.Errorf("Invalid rangefunc, did not accept: %s", v)
	}
}

func TestRangeOR(t *testing.T) {
	tests := []struct {
		v Version
		b bool
	}{
		{MustParse("1.2.0"), true},
		{MustParse("1.2.2"), false},
		{MustParse("1.2.4"), true},
	}
	v1 := MustParse("1.2.1")
	v2 := MustParse("1.2.3")
	rf1 := Range(func(v Version) bool {
		return v.LT(v1)
	})
	rf2 := Range(func(v Version) bool {
		return v.GT(v2)
	})
	rf := rf1.OR(rf2)
	for _, tc := range tests {
		if r := rf(tc.v); r != tc.b {
			t.Errorf("Invalid for case %q: Expected %t, got %t", tc.v, tc.b, r)
		}
	}
}

func TestParseRange(t *testing.T) {
	type tv struct {
		v string
		b bool
	}
	tests := []struct {
		i string
		t []tv
	}{
		// Simple expressions
		{">1.2.3", []tv{
			{"1.2.2", false},
			{"1.2.3", false},
			{"1.2.4", true},
		}},
		{">=1.2.3", []tv{
			{"1.2.3", true},
			{"1.2.4", true},
			{"1.2.2", false},
		}},
		{"<1.2.3", []tv{
			{"1.2.2", true},
			{"1.2.3", false},
			{"1.2.4", false},
		}},
		{"<=1.2.3", []tv{
			{"1.2.2", true},
			{"1.2.3", true},
			{"1.2.4", false},
		}},
		{"1.2.3", []tv{
			{"1.2.2", false},
			{"1.2.3", true},
			{"1.2.4", false},
		}},
		{"=1.2.3", []tv{
			{"1.2.2", false},
			{"1.2.3", true},
			{"1.2.4", false},
		}},
		{"==1.2.3", []tv{
			{"1.2.2", false},
			{"1.2.3", true},
			{"1.2.4", false},
		}},
		{"!=1.2.3", []tv{
			{"1.2.2", true},
			{"1.2.3", false},
			{"1.2.4", true},
		}},
		{"!1.2.3", []tv{
			{"1.2.2", true},
			{"1.2.3", false},
			{"1.2.4", true},
		}},
		// Simple Expression errors
		{">>1.2.3", nil},
		{"!1.2.3", nil},
		{"1.0", nil},
		{"string", nil},
		{"", nil},
		{"fo.ob.ar.x", nil},
		// AND Expressions
		{">1.2.2 <1.2.4", []tv{
			{"1.2.2", false},
			{"1.2.3", true},
			{"1.2.4", false},
		}},
		{"<1.2.2 <1.2.4", []tv{
			{"1.2.1", true},
			{"1.2.2", false},
			{"1.2.3", false},
			{"1.2.4", false},
		}},
		{">1.2.2 <1.2.5 !=1.2.4", []tv{
			{"1.2.2", false},
			{"1.2.3", true},
			{"1.2.4", false},
			{"1.2.5", false},
		}},
		{">1.2.2 <1.2.5 !1.2.4", []tv{
			{"1.2.2", false},
			{"1.2.3", true},
			{"1.2.4", false},
			{"1.2.5", false},
		}},
		// OR Expressions
		{">1.2.2 || <1.2.4", []tv{
			{"1.2.2", true},
			{"1.2.3", true},
			{"1.2.4", true},
		}},
		{"<1.2.2 || >1.2.4", []tv{
			{"1.2.2", false},
			{"1.2.3", false},
			{"1.2.4", false},
		}},
		// Wildcard expressions
		{">1.x", []tv{
			{"0.1.9", false},
			{"1.2.6", false},
			{"1.9.0", false},
			{"2.0.0", true},
		}},
		{">1.2.x", []tv{
			{"1.1.9", false},
			{"1.2.6", false},
			{"1.3.0", true},
		}},
		// Combined Expressions
		{">1.2.2 <1.2.4 || >=2.0.0", []tv{
			{"1.2.2", false},
			{"1.2.3", true},
			{"1.2.4", false},
			{"2.0.0", true},
			{"2.0.1", true},
		}},
		{"1.x || >=2.0.x <2.2.x", []tv{
			{"0.9.2", false},
			{"1.2.2", true},
			{"2.0.0", true},
			{"2.1.8", true},
			{"2.2.0", false},
		}},
		{">1.2.2 <1.2.4 || >=2.0.0 <3.0.0", []tv{
			{"1.2.2", false},
			{"1.2.3", true},
			{"1.2.4", false},
			{"2.0.0", true},
			{"2.0.1", true},
			{"2.9.9", true},
			{"3.0.0", false},
		}},
	}

	for _, tc := range tests {
		r, err := ParseRange(tc.i)
		if err != nil && tc.t != nil {
			t.Errorf("Error parsing range %q: %s", tc.i, err)
			continue
		}
		for _, tvc := range tc.t {
			v := MustParse(tvc.v)
			if res := r(v); res != tvc.b {
				t.Errorf("Invalid for case %q matching %q: Expected %t, got: %t", tc.i, tvc.v, tvc.b, res)
			}
		}

	}
}

func TestMustParseRange(t *testing.T) {
	testCase := ">1.2.2 <1.2.4 || >=2.0.0 <3.0.0"
	r := MustParseRange(testCase)
	if !r(MustParse("1.2.3")) {
		t.Errorf("Unexpected range behavior on MustParseRange")
	}
}

func TestMustParseRange_panic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Errorf("Should have panicked")
		}
	}()
	_ = MustParseRange("invalid version")
}

func BenchmarkRangeParseSimple(b *testing.B) {
	const VERSION = ">1.0.0"
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		ParseRange(VERSION)
	}
}

func BenchmarkRangeParseAverage(b *testing.B) {
	const VERSION = ">=1.0.0 <2.0.0"
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		ParseRange(VERSION)
	}
}

func BenchmarkRangeParseComplex(b *testing.B) {
	const VERSION = ">=1.0.0 <2.0.0 || >=3.0.1 <4.0.0 !=3.0.3 || >=5.0.0"
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		ParseRange(VERSION)
	}
}

func BenchmarkRangeMatchSimple(b *testing.B) {
	const VERSION = ">1.0.0"
	r, _ := ParseRange(VERSION)
	v := MustParse("2.0.0")
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		r(v)
	}
}

func BenchmarkRangeMatchAverage(b *testing.B) {
	const VERSION = ">=1.0.0 <2.0.0"
	r, _ := ParseRange(VERSION)
	v := MustParse("1.2.3")
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		r(v)
	}
}

func BenchmarkRangeMatchComplex(b *testing.B) {
	const VERSION = ">=1.0.0 <2.0.0 || >=3.0.1 <4.0.0 !=3.0.3 || >=5.0.0"
	r, _ := ParseRange(VERSION)
	v := MustParse("5.0.1")
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		r(v)
	}
}
