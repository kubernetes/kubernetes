package set

import "testing"

func TestSelectString(t *testing.T) {
	testCases := []struct {
		s        string
		spec     string
		expected bool
	}{
		{"abcd", "*", true},
		{"abcd", "dcba", false},
		{"abcd", "***d", true},
		{"abcd", "*bcd", true},
		{"abcd", "*bc", false},
		{"abcd", "*cd", true},
		{"abcd", "a*d", true},
		{"abcd", "a*cd", true},
		{"abcde", "a*c*e", false},
	}
	for _, item := range testCases {
		if actual := selectString(item.s, item.spec); actual != item.expected {
			t.Errorf("Expected: %v, got: %v for spec: %s, s: %s ", item.expected, actual, item.spec, item.s)
		}
	}
}
