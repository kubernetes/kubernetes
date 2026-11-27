/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package content

import (
	"regexp"
	"slices"
	"strings"
	"testing"
)

func TestIsDNS1123Label(t *testing.T) {
	goodValues := []string{
		"a", "ab", "abc", "a1", "a-1", "a--1--2--b",
		"0", "01", "012", "1a", "1-a", "1--a--b--2",
		strings.Repeat("a", 63),
	}
	for _, val := range goodValues {
		if msgs := IsDNS1123Label(val); len(msgs) != 0 {
			t.Errorf("expected true for '%s': %v", val, msgs)
		}
	}

	badValues := []string{
		"", "A", "ABC", "aBc", "A1", "A-1", "1-A",
		"-", "a-", "-a", "1-", "-1",
		"_", "a_", "_a", "a_b", "1_", "_1", "1_2",
		".", "a.", ".a", "a.b", "1.", ".1", "1.2",
		" ", "a ", " a", "a b", "1 ", " 1", "1 2",
		strings.Repeat("a", 64),
	}
	for _, val := range badValues {
		if msgs := IsDNS1123Label(val); len(msgs) == 0 {
			t.Errorf("expected false for '%s'", val)
		}
	}
}

func TestIsDNS1123Subdomain(t *testing.T) {
	goodValues := []string{
		"a", "ab", "abc", "a1", "a-1", "a--1--2--b",
		"0", "01", "012", "1a", "1-a", "1--a--b--2",
		"a.a", "ab.a", "abc.a", "a1.a", "a-1.a", "a--1--2--b.a",
		"a.1", "ab.1", "abc.1", "a1.1", "a-1.1", "a--1--2--b.1",
		"0.a", "01.a", "012.a", "1a.a", "1-a.a", "1--a--b--2",
		"0.1", "01.1", "012.1", "1a.1", "1-a.1", "1--a--b--2.1",
		"a.b.c.d.e", "aa.bb.cc.dd.ee", "1.2.3.4.5", "11.22.33.44.55",
		strings.Repeat("a", 253),
	}
	for _, val := range goodValues {
		if msgs := IsDNS1123Subdomain(val); len(msgs) != 0 {
			t.Errorf("expected true for '%s': %v", val, msgs)
		}
	}

	badValues := []string{
		"", "A", "ABC", "aBc", "A1", "A-1", "1-A",
		"-", "a-", "-a", "1-", "-1",
		"_", "a_", "_a", "a_b", "1_", "_1", "1_2",
		".", "a.", ".a", "a..b", "1.", ".1", "1..2",
		" ", "a ", " a", "a b", "1 ", " 1", "1 2",
		"A.a", "aB.a", "ab.A", "A1.a", "a1.A",
		"A.1", "aB.1", "A1.1", "1A.1",
		"0.A", "01.A", "012.A", "1A.a", "1a.A",
		"A.B.C.D.E", "AA.BB.CC.DD.EE", "a.B.c.d.e", "aa.bB.cc.dd.ee",
		"a@b", "a,b", "a_b", "a;b",
		"a:b", "a%b", "a?b", "a$b",
		strings.Repeat("a", 254),
	}
	for _, val := range badValues {
		if msgs := IsDNS1123Subdomain(val); len(msgs) == 0 {
			t.Errorf("expected false for '%s'", val)
		}
	}
}

func TestIsDNS1123SubdomainCaseless(t *testing.T) {
	goodValues := []string{
		"a", "ab", "abc", "a1", "a-1", "a--1--2--b",
		"0", "01", "012", "1a", "1-a", "1--a--b--2",
		"a.a", "ab.a", "abc.a", "a1.a", "a-1.a", "a--1--2--b.a",
		"a.1", "ab.1", "abc.1", "a1.1", "a-1.1", "a--1--2--b.1",
		"0.a", "01.a", "012.a", "1a.a", "1-a.a", "1--a--b--2",
		"0.1", "01.1", "012.1", "1a.1", "1-a.1", "1--a--b--2.1",
		"a.b.c.d.e", "aa.bb.cc.dd.ee", "1.2.3.4.5", "11.22.33.44.55",
		"A", "AB", "ABC", "A1", "A-1",
		"A.A", "AB.A", "ABC.A", "A1.A", "A-1.A",
		"A.B.C.D.E", "AA.BB.CC.DD.EE",
		"a.B.c.d.e", "aa.bB.cc.dd.ee",
		strings.Repeat("a", 253),
		strings.Repeat("A", 253),
	}
	for _, val := range goodValues {
		if msgs := IsDNS1123SubdomainCaseless(val); len(msgs) != 0 {
			t.Errorf("expected true for '%s': %v", val, msgs)
		}
	}

	badValues := []string{
		"",
		"-", "a-", "-a", "1-", "-1",
		"_", "a_", "_a", "a_b", "1_", "_1", "1_2",
		".", "a.", ".a", "a..b", "1.", ".1", "1..2",
		" ", "a ", " a", "a b", "1 ", " 1", "1 2",
		"a@b", "a,b", "a_b", "a;b",
		"a:b", "a%b", "a?b", "a$b",
		strings.Repeat("a", 254),
	}
	for _, val := range badValues {
		if msgs := IsDNS1123SubdomainCaseless(val); len(msgs) == 0 {
			t.Errorf("expected false for '%s'", val)
		}
	}
}

func FuzzIsDNS1123Label(f *testing.F) {
	refLabel := regexp.MustCompile("^" + dns1123LabelFmt + "$")
	refSubdomain := regexp.MustCompile("^" + dns1123SubdomainFmt + "$")
	for _, seed := range []string{"", "a", "abc-123", "a.b", "A", "-a", "a-", strings.Repeat("a", 63), strings.Repeat("a", 64), "a\u00e9b"} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, value string) {
		var want []string
		if len(value) > DNS1123LabelMaxLength {
			want = append(want, MaxLenError(DNS1123LabelMaxLength))
		}
		if !refLabel.MatchString(value) {
			if refSubdomain.MatchString(value) {
				want = append(want, "must not contain dots")
			} else {
				want = append(want, RegexError(dns1123LabelErrMsg, dns1123LabelFmt, "my-name", "123-abc"))
			}
		}
		got := IsDNS1123Label(value)
		if !slices.Equal(got, want) {
			t.Fatalf("IsDNS1123Label(%q) = %v, want %v", value, got, want)
		}
	})
}

func FuzzIsDNS1123Subdomain(f *testing.F) {
	refSubdomain := regexp.MustCompile("^" + dns1123SubdomainFmt + "$")
	refSubdomainCaseless := regexp.MustCompile("^" + dns1123SubdomainFmtCaseless + "$")
	for _, seed := range []string{"", "a", "example.com", "Example.COM", "a..b", ".a", "a.", "a-1.b-2", strings.Repeat("a", 253), strings.Repeat("a", 254), "a.\u00e9"} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, value string) {
		var want []string
		if len(value) > DNS1123SubdomainMaxLength {
			want = append(want, MaxLenError(DNS1123SubdomainMaxLength))
		}
		if !refSubdomain.MatchString(value) {
			want = append(want, RegexError(dns1123SubdomainErrorMsg, dns1123SubdomainFmt, "example.com"))
		}
		if got := IsDNS1123Subdomain(value); !slices.Equal(got, want) {
			t.Fatalf("IsDNS1123Subdomain(%q) = %v, want %v", value, got, want)
		}

		var wantCaseless []string
		if len(value) > DNS1123SubdomainMaxLength {
			wantCaseless = append(wantCaseless, MaxLenError(DNS1123SubdomainMaxLength))
		}
		if !refSubdomainCaseless.MatchString(value) {
			wantCaseless = append(wantCaseless, RegexError(dns1123SubdomainCaselessErrorMsg, dns1123SubdomainFmt, "Example.com"))
		}
		if got := IsDNS1123SubdomainCaseless(value); !slices.Equal(got, wantCaseless) {
			t.Fatalf("IsDNS1123SubdomainCaseless(%q) = %v, want %v", value, got, wantCaseless)
		}
	})
}

func BenchmarkIsDNS1123Label(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = IsDNS1123Label("my-node-pool-us-central1-a-001")
	}
}

func BenchmarkIsDNS1123Subdomain(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = IsDNS1123Subdomain("topology.kubernetes.io")
	}
}
