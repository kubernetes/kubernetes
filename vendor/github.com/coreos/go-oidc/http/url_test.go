package http

import (
	"net/url"
	"testing"
)

func TestParseNonEmptyURL(t *testing.T) {
	tests := []struct {
		u  string
		ok bool
	}{
		{"", false},
		{"http://", false},
		{"example.com", false},
		{"example", false},
		{"http://example", true},
		{"http://example:1234", true},
		{"http://example.com", true},
		{"http://example.com:1234", true},
	}

	for i, tt := range tests {
		u, err := ParseNonEmptyURL(tt.u)
		if err != nil {
			t.Logf("err: %v", err)
			if tt.ok {
				t.Errorf("case %d: unexpected error: %v", i, err)
			} else {
				continue
			}
		}

		if !tt.ok {
			t.Errorf("case %d: expected error but got none", i)
			continue
		}

		uu, err := url.Parse(tt.u)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
			continue
		}

		if uu.String() != u.String() {
			t.Errorf("case %d: incorrect url value, want: %q, got: %q", i, uu.String(), u.String())
		}
	}
}
