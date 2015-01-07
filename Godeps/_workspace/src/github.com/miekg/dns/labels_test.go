package dns

import (
	"testing"
)

func TestCompareDomainName(t *testing.T) {
	s1 := "www.miek.nl."
	s2 := "miek.nl."
	s3 := "www.bla.nl."
	s4 := "nl.www.bla."
	s5 := "nl"
	s6 := "miek.nl"

	if CompareDomainName(s1, s2) != 2 {
		t.Logf("%s with %s should be %d", s1, s2, 2)
		t.Fail()
	}
	if CompareDomainName(s1, s3) != 1 {
		t.Logf("%s with %s should be %d", s1, s3, 1)
		t.Fail()
	}
	if CompareDomainName(s3, s4) != 0 {
		t.Logf("%s with %s should be %d", s3, s4, 0)
		t.Fail()
	}
	// Non qualified tests
	if CompareDomainName(s1, s5) != 1 {
		t.Logf("%s with %s should be %d", s1, s5, 1)
		t.Fail()
	}
	if CompareDomainName(s1, s6) != 2 {
		t.Logf("%s with %s should be %d", s1, s5, 2)
		t.Fail()
	}

	if CompareDomainName(s1, ".") != 0 {
		t.Logf("%s with %s should be %d", s1, s5, 0)
		t.Fail()
	}
	if CompareDomainName(".", ".") != 0 {
		t.Logf("%s with %s should be %d", ".", ".", 0)
		t.Fail()
	}
}

func TestSplit(t *testing.T) {
	splitter := map[string]int{
		"www.miek.nl.":   3,
		"www.miek.nl":    3,
		"www..miek.nl":   4,
		`www\.miek.nl.`:  2,
		`www\\.miek.nl.`: 3,
		".":              0,
		"nl.":            1,
		"nl":             1,
		"com.":           1,
		".com.":          2,
	}
	for s, i := range splitter {
		if x := len(Split(s)); x != i {
			t.Logf("labels should be %d, got %d: %s %v\n", i, x, s, Split(s))
			t.Fail()
		} else {
			t.Logf("%s %v\n", s, Split(s))
		}
	}
}

func TestSplit2(t *testing.T) {
	splitter := map[string][]int{
		"www.miek.nl.": []int{0, 4, 9},
		"www.miek.nl":  []int{0, 4, 9},
		"nl":           []int{0},
	}
	for s, i := range splitter {
		x := Split(s)
		switch len(i) {
		case 1:
			if x[0] != i[0] {
				t.Logf("labels should be %v, got %v: %s\n", i, x, s)
				t.Fail()
			}
		default:
			if x[0] != i[0] || x[1] != i[1] || x[2] != i[2] {
				t.Logf("labels should be %v, got %v: %s\n", i, x, s)
				t.Fail()
			}
		}
	}
}

func TestPrevLabel(t *testing.T) {
	type prev struct {
		string
		int
	}
	prever := map[prev]int{
		prev{"www.miek.nl.", 0}: 12,
		prev{"www.miek.nl.", 1}: 9,
		prev{"www.miek.nl.", 2}: 4,

		prev{"www.miek.nl", 0}: 11,
		prev{"www.miek.nl", 1}: 9,
		prev{"www.miek.nl", 2}: 4,

		prev{"www.miek.nl.", 5}: 0,
		prev{"www.miek.nl", 5}:  0,

		prev{"www.miek.nl.", 3}: 0,
		prev{"www.miek.nl", 3}:  0,
	}
	for s, i := range prever {
		x, ok := PrevLabel(s.string, s.int)
		if i != x {
			t.Logf("label should be %d, got %d, %t: preving %d, %s\n", i, x, ok, s.int, s.string)
			t.Fail()
		}
	}
}

func TestCountLabel(t *testing.T) {
	splitter := map[string]int{
		"www.miek.nl.": 3,
		"www.miek.nl":  3,
		"nl":           1,
		".":            0,
	}
	for s, i := range splitter {
		x := CountLabel(s)
		if x != i {
			t.Logf("CountLabel should have %d, got %d\n", i, x)
			t.Fail()
		}
	}
}

func TestSplitDomainName(t *testing.T) {
	labels := map[string][]string{
		"miek.nl":       []string{"miek", "nl"},
		".":             nil,
		"www.miek.nl.":  []string{"www", "miek", "nl"},
		"www.miek.nl":   []string{"www", "miek", "nl"},
		"www..miek.nl":  []string{"www", "", "miek", "nl"},
		`www\.miek.nl`:  []string{`www\.miek`, "nl"},
		`www\\.miek.nl`: []string{`www\\`, "miek", "nl"},
	}
domainLoop:
	for domain, splits := range labels {
		parts := SplitDomainName(domain)
		if len(parts) != len(splits) {
			t.Logf("SplitDomainName returned %v for %s, expected %v", parts, domain, splits)
			t.Fail()
			continue domainLoop
		}
		for i := range parts {
			if parts[i] != splits[i] {
				t.Logf("SplitDomainName returned %v for %s, expected %v", parts, domain, splits)
				t.Fail()
				continue domainLoop
			}
		}
	}
}

func TestIsDomainName(t *testing.T) {
	type ret struct {
		ok  bool
		lab int
	}
	names := map[string]*ret{
		"..":               &ret{false, 1},
		"@.":               &ret{true, 1},
		"www.example.com":  &ret{true, 3},
		"www.e%ample.com":  &ret{true, 3},
		"www.example.com.": &ret{true, 3},
		"mi\\k.nl.":        &ret{true, 2},
		"mi\\k.nl":         &ret{true, 2},
	}
	for d, ok := range names {
		l, k := IsDomainName(d)
		if ok.ok != k || ok.lab != l {
			t.Logf(" got %v %d for %s ", k, l, d)
			t.Logf("have %v %d for %s ", ok.ok, ok.lab, d)
			t.Fail()
		}
	}
}

func BenchmarkSplitLabels(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Split("www.example.com")
	}
}

func BenchmarkLenLabels(b *testing.B) {
	for i := 0; i < b.N; i++ {
		CountLabel("www.example.com")
	}
}

func BenchmarkCompareLabels(b *testing.B) {
	for i := 0; i < b.N; i++ {
		CompareDomainName("www.example.com", "aa.example.com")
	}
}

func BenchmarkIsSubDomain(b *testing.B) {
	for i := 0; i < b.N; i++ {
		IsSubDomain("www.example.com", "aa.example.com")
		IsSubDomain("example.com", "aa.example.com")
		IsSubDomain("miek.nl", "aa.example.com")
	}
}
