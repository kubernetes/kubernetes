package netutil

import (
	"net/url"
	"testing"
)

func Test_CanonicalAddr(t *testing.T) {
	tests := []struct {
		expected string
		url      *url.URL
	}{
		{
			expected: "foo.com:8080",
			url:      &url.URL{Host: "foo.com:8080"},
		},
		{
			expected: "bar.com:80",
			url:      &url.URL{Host: "bar.com", Scheme: "http"},
		},
		{
			expected: "baz.com:443",
			url:      &url.URL{Host: "baz.com", Scheme: "https"},
		},
	}

	for _, test := range tests {
		result := CanonicalAddr(test.url)
		if result != test.expected {
			t.Errorf("expect %s, got %s", test.expected, result)
		}
	}
}
