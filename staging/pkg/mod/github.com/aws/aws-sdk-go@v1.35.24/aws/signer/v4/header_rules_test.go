package v4

import (
	"testing"
)

func TestRuleCheckWhitelist(t *testing.T) {
	w := whitelist{
		mapRule{
			"Cache-Control": struct{}{},
		},
	}

	if !w.IsValid("Cache-Control") {
		t.Error("expected true value")
	}
	if w.IsValid("Cache-") {
		t.Error("expected false value")
	}
}

func TestRuleCheckBlacklist(t *testing.T) {
	b := blacklist{
		mapRule{
			"Cache-Control": struct{}{},
		},
	}

	if b.IsValid("Cache-Control") {
		t.Error("expected false value")
	}
	if !b.IsValid("Cache-") {
		t.Error("expected true value")
	}
}

func TestRuleCheckPattern(t *testing.T) {
	p := patterns{"X-Amz-Meta-"}

	if !p.IsValid("X-Amz-Meta-") {
		t.Error("expected true value")
	}
	if !p.IsValid("X-Amz-Meta-Star") {
		t.Error("expected true value")
	}
	if p.IsValid("Cache-") {
		t.Error("expected false value")
	}
}

func TestRuleComplexWhitelist(t *testing.T) {
	w := rules{
		whitelist{
			mapRule{
				"Cache-Control": struct{}{},
			},
		},
		patterns{"X-Amz-Meta-"},
	}

	r := rules{
		inclusiveRules{patterns{"X-Amz-"}, blacklist{w}},
	}

	if !r.IsValid("X-Amz-Blah") {
		t.Error("expected true value")
	}
	if r.IsValid("X-Amz-Meta-") {
		t.Error("expected false value")
	}
	if r.IsValid("X-Amz-Meta-Star") {
		t.Error("expected false value")
	}
	if r.IsValid("Cache-Control") {
		t.Error("expected false value")
	}
}
