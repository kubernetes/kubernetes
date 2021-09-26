// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package publicsuffix

import (
	"sort"
	"strings"
	"testing"
)

func TestNodeLabel(t *testing.T) {
	for i, want := range nodeLabels {
		got := nodeLabel(uint32(i))
		if got != want {
			t.Errorf("%d: got %q, want %q", i, got, want)
		}
	}
}

func TestFind(t *testing.T) {
	testCases := []string{
		"",
		"a",
		"a0",
		"aaaa",
		"ao",
		"ap",
		"ar",
		"aro",
		"arp",
		"arpa",
		"arpaa",
		"arpb",
		"az",
		"b",
		"b0",
		"ba",
		"z",
		"zu",
		"zv",
		"zw",
		"zx",
		"zy",
		"zz",
		"zzzz",
	}
	for _, tc := range testCases {
		got := find(tc, 0, numTLD)
		want := notFound
		for i := uint32(0); i < numTLD; i++ {
			if tc == nodeLabel(i) {
				want = i
				break
			}
		}
		if got != want {
			t.Errorf("%q: got %d, want %d", tc, got, want)
		}
	}
}

func TestICANN(t *testing.T) {
	testCases := map[string]bool{
		"foo.org":            true,
		"foo.co.uk":          true,
		"foo.dyndns.org":     false,
		"foo.go.dyndns.org":  false,
		"foo.blogspot.co.uk": false,
		"foo.intranet":       false,
	}
	for domain, want := range testCases {
		_, got := PublicSuffix(domain)
		if got != want {
			t.Errorf("%q: got %v, want %v", domain, got, want)
		}
	}
}

var publicSuffixTestCases = []struct {
	domain    string
	wantPS    string
	wantICANN bool
}{
	// Empty string.
	{"", "", false},

	// The .ao rules are:
	// ao
	// ed.ao
	// gv.ao
	// og.ao
	// co.ao
	// pb.ao
	// it.ao
	{"ao", "ao", true},
	{"www.ao", "ao", true},
	{"pb.ao", "pb.ao", true},
	{"www.pb.ao", "pb.ao", true},
	{"www.xxx.yyy.zzz.pb.ao", "pb.ao", true},

	// The .ar rules are:
	// ar
	// com.ar
	// edu.ar
	// gob.ar
	// gov.ar
	// int.ar
	// mil.ar
	// net.ar
	// org.ar
	// tur.ar
	// blogspot.com.ar (in the PRIVATE DOMAIN section).
	{"ar", "ar", true},
	{"www.ar", "ar", true},
	{"nic.ar", "ar", true},
	{"www.nic.ar", "ar", true},
	{"com.ar", "com.ar", true},
	{"www.com.ar", "com.ar", true},
	{"blogspot.com.ar", "blogspot.com.ar", false},                 // PRIVATE DOMAIN.
	{"www.blogspot.com.ar", "blogspot.com.ar", false},             // PRIVATE DOMAIN.
	{"www.xxx.yyy.zzz.blogspot.com.ar", "blogspot.com.ar", false}, // PRIVATE DOMAIN.
	{"logspot.com.ar", "com.ar", true},
	{"zlogspot.com.ar", "com.ar", true},
	{"zblogspot.com.ar", "com.ar", true},

	// The .arpa rules are:
	// arpa
	// e164.arpa
	// in-addr.arpa
	// ip6.arpa
	// iris.arpa
	// uri.arpa
	// urn.arpa
	{"arpa", "arpa", true},
	{"www.arpa", "arpa", true},
	{"urn.arpa", "urn.arpa", true},
	{"www.urn.arpa", "urn.arpa", true},
	{"www.xxx.yyy.zzz.urn.arpa", "urn.arpa", true},

	// The relevant {kobe,kyoto}.jp rules are:
	// jp
	// *.kobe.jp
	// !city.kobe.jp
	// kyoto.jp
	// ide.kyoto.jp
	{"jp", "jp", true},
	{"kobe.jp", "jp", true},
	{"c.kobe.jp", "c.kobe.jp", true},
	{"b.c.kobe.jp", "c.kobe.jp", true},
	{"a.b.c.kobe.jp", "c.kobe.jp", true},
	{"city.kobe.jp", "kobe.jp", true},
	{"www.city.kobe.jp", "kobe.jp", true},
	{"kyoto.jp", "kyoto.jp", true},
	{"test.kyoto.jp", "kyoto.jp", true},
	{"ide.kyoto.jp", "ide.kyoto.jp", true},
	{"b.ide.kyoto.jp", "ide.kyoto.jp", true},
	{"a.b.ide.kyoto.jp", "ide.kyoto.jp", true},

	// The .tw rules are:
	// tw
	// edu.tw
	// gov.tw
	// mil.tw
	// com.tw
	// net.tw
	// org.tw
	// idv.tw
	// game.tw
	// ebiz.tw
	// club.tw
	// 網路.tw (xn--zf0ao64a.tw)
	// 組織.tw (xn--uc0atv.tw)
	// 商業.tw (xn--czrw28b.tw)
	// blogspot.tw
	{"tw", "tw", true},
	{"aaa.tw", "tw", true},
	{"www.aaa.tw", "tw", true},
	{"xn--czrw28b.aaa.tw", "tw", true},
	{"edu.tw", "edu.tw", true},
	{"www.edu.tw", "edu.tw", true},
	{"xn--czrw28b.edu.tw", "edu.tw", true},
	{"xn--czrw28b.tw", "xn--czrw28b.tw", true},
	{"www.xn--czrw28b.tw", "xn--czrw28b.tw", true},
	{"xn--uc0atv.xn--czrw28b.tw", "xn--czrw28b.tw", true},
	{"xn--kpry57d.tw", "tw", true},

	// The .uk rules are:
	// uk
	// ac.uk
	// co.uk
	// gov.uk
	// ltd.uk
	// me.uk
	// net.uk
	// nhs.uk
	// org.uk
	// plc.uk
	// police.uk
	// *.sch.uk
	// blogspot.co.uk (in the PRIVATE DOMAIN section).
	{"uk", "uk", true},
	{"aaa.uk", "uk", true},
	{"www.aaa.uk", "uk", true},
	{"mod.uk", "uk", true},
	{"www.mod.uk", "uk", true},
	{"sch.uk", "uk", true},
	{"mod.sch.uk", "mod.sch.uk", true},
	{"www.sch.uk", "www.sch.uk", true},
	{"co.uk", "co.uk", true},
	{"www.co.uk", "co.uk", true},
	{"blogspot.co.uk", "blogspot.co.uk", false}, // PRIVATE DOMAIN.
	{"blogspot.nic.uk", "uk", true},
	{"blogspot.sch.uk", "blogspot.sch.uk", true},

	// The .рф rules are
	// рф (xn--p1ai)
	{"xn--p1ai", "xn--p1ai", true},
	{"aaa.xn--p1ai", "xn--p1ai", true},
	{"www.xxx.yyy.xn--p1ai", "xn--p1ai", true},

	// The .bd rules are:
	// *.bd
	{"bd", "bd", false}, // The catch-all "*" rule is not in the ICANN DOMAIN section. See footnote (†).
	{"www.bd", "www.bd", true},
	{"xxx.www.bd", "www.bd", true},
	{"zzz.bd", "zzz.bd", true},
	{"www.zzz.bd", "zzz.bd", true},
	{"www.xxx.yyy.zzz.bd", "zzz.bd", true},

	// The .ck rules are:
	// *.ck
	// !www.ck
	{"ck", "ck", false}, // The catch-all "*" rule is not in the ICANN DOMAIN section. See footnote (†).
	{"www.ck", "ck", true},
	{"xxx.www.ck", "ck", true},
	{"zzz.ck", "zzz.ck", true},
	{"www.zzz.ck", "zzz.ck", true},
	{"www.xxx.yyy.zzz.ck", "zzz.ck", true},

	// The .myjino.ru rules (in the PRIVATE DOMAIN section) are:
	// myjino.ru
	// *.hosting.myjino.ru
	// *.landing.myjino.ru
	// *.spectrum.myjino.ru
	// *.vps.myjino.ru
	{"myjino.ru", "myjino.ru", false},
	{"aaa.myjino.ru", "myjino.ru", false},
	{"bbb.ccc.myjino.ru", "myjino.ru", false},
	{"hosting.ddd.myjino.ru", "myjino.ru", false},
	{"landing.myjino.ru", "myjino.ru", false},
	{"www.landing.myjino.ru", "www.landing.myjino.ru", false},
	{"spectrum.vps.myjino.ru", "spectrum.vps.myjino.ru", false},

	// The .uberspace.de rules (in the PRIVATE DOMAIN section) are:
	// *.uberspace.de
	{"uberspace.de", "de", true}, // "de" is in the ICANN DOMAIN section. See footnote (†).
	{"aaa.uberspace.de", "aaa.uberspace.de", false},
	{"bbb.ccc.uberspace.de", "ccc.uberspace.de", false},

	// There are no .nosuchtld rules.
	{"nosuchtld", "nosuchtld", false},
	{"foo.nosuchtld", "nosuchtld", false},
	{"bar.foo.nosuchtld", "nosuchtld", false},

	// (†) There is some disagreement on how wildcards behave: what should the
	// public suffix of "platform.sh" be when both "*.platform.sh" and "sh" is
	// in the PSL, but "platform.sh" is not? Two possible answers are
	// "platform.sh" and "sh", there are valid arguments for either behavior,
	// and different browsers have implemented different behaviors.
	//
	// This implementation, Go's golang.org/x/net/publicsuffix, returns "sh",
	// the same as a literal interpretation of the "Formal Algorithm" section
	// of https://publicsuffix.org/list/
	//
	// Together, the TestPublicSuffix and TestSlowPublicSuffix tests check that
	// the Go implementation (func PublicSuffix in list.go) and the literal
	// interpretation (func slowPublicSuffix in list_test.go) produce the same
	// (golden) results on every test case in this publicSuffixTestCases slice,
	// including some "platform.sh" style cases.
	//
	// More discussion of "the platform.sh problem" is at:
	//  - https://github.com/publicsuffix/list/issues/694
	//  - https://bugzilla.mozilla.org/show_bug.cgi?id=1124625#c6
	//  - https://wiki.mozilla.org/Public_Suffix_List/platform.sh_Problem
}

func BenchmarkPublicSuffix(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for _, tc := range publicSuffixTestCases {
			List.PublicSuffix(tc.domain)
		}
	}
}

func TestPublicSuffix(t *testing.T) {
	for _, tc := range publicSuffixTestCases {
		gotPS, gotICANN := PublicSuffix(tc.domain)
		if gotPS != tc.wantPS || gotICANN != tc.wantICANN {
			t.Errorf("%q: got (%q, %t), want (%q, %t)", tc.domain, gotPS, gotICANN, tc.wantPS, tc.wantICANN)
		}
	}
}

func TestSlowPublicSuffix(t *testing.T) {
	for _, tc := range publicSuffixTestCases {
		gotPS, gotICANN := slowPublicSuffix(tc.domain)
		if gotPS != tc.wantPS || gotICANN != tc.wantICANN {
			t.Errorf("%q: got (%q, %t), want (%q, %t)", tc.domain, gotPS, gotICANN, tc.wantPS, tc.wantICANN)
		}
	}
}

func TestNumICANNRules(t *testing.T) {
	if numICANNRules <= 0 {
		t.Fatal("no ICANN rules")
	}
	if numICANNRules >= len(rules) {
		t.Fatal("no Private rules")
	}
	// Check the last ICANN and first Private rules. If the underlying public
	// suffix list changes, we may need to update these hard-coded checks.
	if got, want := rules[numICANNRules-1], "zuerich"; got != want {
		t.Errorf("last ICANN rule: got %q, wawnt %q", got, want)
	}
	if got, want := rules[numICANNRules], "cc.ua"; got != want {
		t.Errorf("first Private rule: got %q, wawnt %q", got, want)
	}
}

type slowPublicSuffixRule struct {
	ruleParts []string
	icann     bool
}

// slowPublicSuffix implements the canonical (but O(number of rules)) public
// suffix algorithm described at http://publicsuffix.org/list/.
//
// 1. Match domain against all rules and take note of the matching ones.
// 2. If no rules match, the prevailing rule is "*".
// 3. If more than one rule matches, the prevailing rule is the one which is an exception rule.
// 4. If there is no matching exception rule, the prevailing rule is the one with the most labels.
// 5. If the prevailing rule is a exception rule, modify it by removing the leftmost label.
// 6. The public suffix is the set of labels from the domain which directly match the labels of the prevailing rule (joined by dots).
// 7. The registered or registrable domain is the public suffix plus one additional label.
//
// This function returns the public suffix, not the registrable domain, and so
// it stops after step 6.
func slowPublicSuffix(domain string) (string, bool) {
	match := func(rulePart, domainPart string) bool {
		switch rulePart[0] {
		case '*':
			return true
		case '!':
			return rulePart[1:] == domainPart
		}
		return rulePart == domainPart
	}

	domainParts := strings.Split(domain, ".")
	var matchingRules []slowPublicSuffixRule

loop:
	for i, rule := range rules {
		ruleParts := strings.Split(rule, ".")
		if len(domainParts) < len(ruleParts) {
			continue
		}
		for i := range ruleParts {
			rulePart := ruleParts[len(ruleParts)-1-i]
			domainPart := domainParts[len(domainParts)-1-i]
			if !match(rulePart, domainPart) {
				continue loop
			}
		}
		matchingRules = append(matchingRules, slowPublicSuffixRule{
			ruleParts: ruleParts,
			icann:     i < numICANNRules,
		})
	}
	if len(matchingRules) == 0 {
		matchingRules = append(matchingRules, slowPublicSuffixRule{
			ruleParts: []string{"*"},
			icann:     false,
		})
	} else {
		sort.Sort(byPriority(matchingRules))
	}

	prevailing := matchingRules[0]
	if prevailing.ruleParts[0][0] == '!' {
		prevailing.ruleParts = prevailing.ruleParts[1:]
	}
	if prevailing.ruleParts[0][0] == '*' {
		replaced := domainParts[len(domainParts)-len(prevailing.ruleParts)]
		prevailing.ruleParts = append([]string{replaced}, prevailing.ruleParts[1:]...)
	}
	return strings.Join(prevailing.ruleParts, "."), prevailing.icann
}

type byPriority []slowPublicSuffixRule

func (b byPriority) Len() int      { return len(b) }
func (b byPriority) Swap(i, j int) { b[i], b[j] = b[j], b[i] }
func (b byPriority) Less(i, j int) bool {
	if b[i].ruleParts[0][0] == '!' {
		return true
	}
	if b[j].ruleParts[0][0] == '!' {
		return false
	}
	return len(b[i].ruleParts) > len(b[j].ruleParts)
}

// eTLDPlusOneTestCases come from
// https://github.com/publicsuffix/list/blob/master/tests/test_psl.txt
var eTLDPlusOneTestCases = []struct {
	domain, want string
}{
	// Empty input.
	{"", ""},
	// Unlisted TLD.
	{"example", ""},
	{"example.example", "example.example"},
	{"b.example.example", "example.example"},
	{"a.b.example.example", "example.example"},
	// TLD with only 1 rule.
	{"biz", ""},
	{"domain.biz", "domain.biz"},
	{"b.domain.biz", "domain.biz"},
	{"a.b.domain.biz", "domain.biz"},
	// TLD with some 2-level rules.
	{"com", ""},
	{"example.com", "example.com"},
	{"b.example.com", "example.com"},
	{"a.b.example.com", "example.com"},
	{"uk.com", ""},
	{"example.uk.com", "example.uk.com"},
	{"b.example.uk.com", "example.uk.com"},
	{"a.b.example.uk.com", "example.uk.com"},
	{"test.ac", "test.ac"},
	// TLD with only 1 (wildcard) rule.
	{"mm", ""},
	{"c.mm", ""},
	{"b.c.mm", "b.c.mm"},
	{"a.b.c.mm", "b.c.mm"},
	// More complex TLD.
	{"jp", ""},
	{"test.jp", "test.jp"},
	{"www.test.jp", "test.jp"},
	{"ac.jp", ""},
	{"test.ac.jp", "test.ac.jp"},
	{"www.test.ac.jp", "test.ac.jp"},
	{"kyoto.jp", ""},
	{"test.kyoto.jp", "test.kyoto.jp"},
	{"ide.kyoto.jp", ""},
	{"b.ide.kyoto.jp", "b.ide.kyoto.jp"},
	{"a.b.ide.kyoto.jp", "b.ide.kyoto.jp"},
	{"c.kobe.jp", ""},
	{"b.c.kobe.jp", "b.c.kobe.jp"},
	{"a.b.c.kobe.jp", "b.c.kobe.jp"},
	{"city.kobe.jp", "city.kobe.jp"},
	{"www.city.kobe.jp", "city.kobe.jp"},
	// TLD with a wildcard rule and exceptions.
	{"ck", ""},
	{"test.ck", ""},
	{"b.test.ck", "b.test.ck"},
	{"a.b.test.ck", "b.test.ck"},
	{"www.ck", "www.ck"},
	{"www.www.ck", "www.ck"},
	// US K12.
	{"us", ""},
	{"test.us", "test.us"},
	{"www.test.us", "test.us"},
	{"ak.us", ""},
	{"test.ak.us", "test.ak.us"},
	{"www.test.ak.us", "test.ak.us"},
	{"k12.ak.us", ""},
	{"test.k12.ak.us", "test.k12.ak.us"},
	{"www.test.k12.ak.us", "test.k12.ak.us"},
	// Punycoded IDN labels
	{"xn--85x722f.com.cn", "xn--85x722f.com.cn"},
	{"xn--85x722f.xn--55qx5d.cn", "xn--85x722f.xn--55qx5d.cn"},
	{"www.xn--85x722f.xn--55qx5d.cn", "xn--85x722f.xn--55qx5d.cn"},
	{"shishi.xn--55qx5d.cn", "shishi.xn--55qx5d.cn"},
	{"xn--55qx5d.cn", ""},
	{"xn--85x722f.xn--fiqs8s", "xn--85x722f.xn--fiqs8s"},
	{"www.xn--85x722f.xn--fiqs8s", "xn--85x722f.xn--fiqs8s"},
	{"shishi.xn--fiqs8s", "shishi.xn--fiqs8s"},
	{"xn--fiqs8s", ""},

	// Invalid input
	{".", ""},
	{"de.", ""},
	{".de", ""},
	{".com.au", ""},
	{"com.au.", ""},
	{"com..au", ""},
}

func TestEffectiveTLDPlusOne(t *testing.T) {
	for _, tc := range eTLDPlusOneTestCases {
		got, _ := EffectiveTLDPlusOne(tc.domain)
		if got != tc.want {
			t.Errorf("%q: got %q, want %q", tc.domain, got, tc.want)
		}
	}
}
