package ldap

import (
	"testing"

	"gopkg.in/asn1-ber.v1"
)

type compileTest struct {
	filterStr  string
	filterType int
}

var testFilters = []compileTest{
	compileTest{filterStr: "(&(sn=Miller)(givenName=Bob))", filterType: FilterAnd},
	compileTest{filterStr: "(|(sn=Miller)(givenName=Bob))", filterType: FilterOr},
	compileTest{filterStr: "(!(sn=Miller))", filterType: FilterNot},
	compileTest{filterStr: "(sn=Miller)", filterType: FilterEqualityMatch},
	compileTest{filterStr: "(sn=Mill*)", filterType: FilterSubstrings},
	compileTest{filterStr: "(sn=*Mill)", filterType: FilterSubstrings},
	compileTest{filterStr: "(sn=*Mill*)", filterType: FilterSubstrings},
	compileTest{filterStr: "(sn=*i*le*)", filterType: FilterSubstrings},
	compileTest{filterStr: "(sn=Mi*l*r)", filterType: FilterSubstrings},
	compileTest{filterStr: "(sn=Mi*le*)", filterType: FilterSubstrings},
	compileTest{filterStr: "(sn=*i*ler)", filterType: FilterSubstrings},
	compileTest{filterStr: "(sn>=Miller)", filterType: FilterGreaterOrEqual},
	compileTest{filterStr: "(sn<=Miller)", filterType: FilterLessOrEqual},
	compileTest{filterStr: "(sn=*)", filterType: FilterPresent},
	compileTest{filterStr: "(sn~=Miller)", filterType: FilterApproxMatch},
	compileTest{filterStr: `(objectGUID='\fc\fe\a3\ab\f9\90N\aaGm\d5I~\d12)`, filterType: FilterEqualityMatch},
	// compileTest{ filterStr: "()", filterType: FilterExtensibleMatch },
}

var testInvalidFilters = []string{
	`(objectGUID=\zz)`,
	`(objectGUID=\a)`,
}

func TestFilter(t *testing.T) {
	// Test Compiler and Decompiler
	for _, i := range testFilters {
		filter, err := CompileFilter(i.filterStr)
		if err != nil {
			t.Errorf("Problem compiling %s - %s", i.filterStr, err.Error())
		} else if filter.Tag != ber.Tag(i.filterType) {
			t.Errorf("%q Expected %q got %q", i.filterStr, FilterMap[uint64(i.filterType)], FilterMap[uint64(filter.Tag)])
		} else {
			o, err := DecompileFilter(filter)
			if err != nil {
				t.Errorf("Problem compiling %s - %s", i.filterStr, err.Error())
			} else if i.filterStr != o {
				t.Errorf("%q expected, got %q", i.filterStr, o)
			}
		}
	}
}

func TestInvalidFilter(t *testing.T) {
	for _, filterStr := range testInvalidFilters {
		if _, err := CompileFilter(filterStr); err == nil {
			t.Errorf("Problem compiling %s - expected err", filterStr)
		}
	}
}

func BenchmarkFilterCompile(b *testing.B) {
	b.StopTimer()
	filters := make([]string, len(testFilters))

	// Test Compiler and Decompiler
	for idx, i := range testFilters {
		filters[idx] = i.filterStr
	}

	maxIdx := len(filters)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		CompileFilter(filters[i%maxIdx])
	}
}

func BenchmarkFilterDecompile(b *testing.B) {
	b.StopTimer()
	filters := make([]*ber.Packet, len(testFilters))

	// Test Compiler and Decompiler
	for idx, i := range testFilters {
		filters[idx], _ = CompileFilter(i.filterStr)
	}

	maxIdx := len(filters)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		DecompileFilter(filters[i%maxIdx])
	}
}
