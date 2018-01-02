package dnsutil

import "testing"

func TestAddOrigin(t *testing.T) {
	var tests = []struct{ e1, e2, expected string }{
		{"@", "example.com", "example.com"},
		{"foo", "example.com", "foo.example.com"},
		{"foo.", "example.com", "foo."},
		{"@", "example.com.", "example.com."},
		{"foo", "example.com.", "foo.example.com."},
		{"foo.", "example.com.", "foo."},
		// Oddball tests:
		// In general origin should not be "" or "." but at least
		// these tests verify we don't crash and will keep results
		// from changing unexpectedly.
		{"*.", "", "*."},
		{"@", "", "@"},
		{"foobar", "", "foobar"},
		{"foobar.", "", "foobar."},
		{"*.", ".", "*."},
		{"@", ".", "."},
		{"foobar", ".", "foobar."},
		{"foobar.", ".", "foobar."},
	}
	for _, test := range tests {
		actual := AddOrigin(test.e1, test.e2)
		if test.expected != actual {
			t.Errorf("AddOrigin(%#v, %#v) expected %#v, go %#v\n", test.e1, test.e2, test.expected, actual)
		}
	}
}

func TestTrimDomainName(t *testing.T) {

	// Basic tests.
	// Try trimming "example.com" and "example.com." from typical use cases.
	var tests_examplecom = []struct{ experiment, expected string }{
		{"foo.example.com", "foo"},
		{"foo.example.com.", "foo"},
		{".foo.example.com", ".foo"},
		{".foo.example.com.", ".foo"},
		{"*.example.com", "*"},
		{"example.com", "@"},
		{"example.com.", "@"},
		{"com.", "com."},
		{"foo.", "foo."},
		{"serverfault.com.", "serverfault.com."},
		{"serverfault.com", "serverfault.com"},
		{".foo.ronco.com", ".foo.ronco.com"},
		{".foo.ronco.com.", ".foo.ronco.com."},
	}
	for _, dom := range []string{"example.com", "example.com."} {
		for i, test := range tests_examplecom {
			actual := TrimDomainName(test.experiment, dom)
			if test.expected != actual {
				t.Errorf("%d TrimDomainName(%#v, %#v): expected (%v) got (%v)\n", i, test.experiment, dom, test.expected, actual)
			}
		}
	}

	// Paranoid tests.
	// These test shouldn't be needed but I was weary of off-by-one errors.
	// In theory, these can't happen because there are no single-letter TLDs,
	// but it is good to exercize the code this way.
	var tests = []struct{ experiment, expected string }{
		{"", "@"},
		{".", "."},
		{"a.b.c.d.e.f.", "a.b.c.d.e"},
		{"b.c.d.e.f.", "b.c.d.e"},
		{"c.d.e.f.", "c.d.e"},
		{"d.e.f.", "d.e"},
		{"e.f.", "e"},
		{"f.", "@"},
		{".a.b.c.d.e.f.", ".a.b.c.d.e"},
		{".b.c.d.e.f.", ".b.c.d.e"},
		{".c.d.e.f.", ".c.d.e"},
		{".d.e.f.", ".d.e"},
		{".e.f.", ".e"},
		{".f.", "@"},
		{"a.b.c.d.e.f", "a.b.c.d.e"},
		{"a.b.c.d.e.", "a.b.c.d.e."},
		{"a.b.c.d.e", "a.b.c.d.e"},
		{"a.b.c.d.", "a.b.c.d."},
		{"a.b.c.d", "a.b.c.d"},
		{"a.b.c.", "a.b.c."},
		{"a.b.c", "a.b.c"},
		{"a.b.", "a.b."},
		{"a.b", "a.b"},
		{"a.", "a."},
		{"a", "a"},
		{".a.b.c.d.e.f", ".a.b.c.d.e"},
		{".a.b.c.d.e.", ".a.b.c.d.e."},
		{".a.b.c.d.e", ".a.b.c.d.e"},
		{".a.b.c.d.", ".a.b.c.d."},
		{".a.b.c.d", ".a.b.c.d"},
		{".a.b.c.", ".a.b.c."},
		{".a.b.c", ".a.b.c"},
		{".a.b.", ".a.b."},
		{".a.b", ".a.b"},
		{".a.", ".a."},
		{".a", ".a"},
	}
	for _, dom := range []string{"f", "f."} {
		for i, test := range tests {
			actual := TrimDomainName(test.experiment, dom)
			if test.expected != actual {
				t.Errorf("%d TrimDomainName(%#v, %#v): expected (%v) got (%v)\n", i, test.experiment, dom, test.expected, actual)
			}
		}
	}

	// Test cases for bugs found in the wild.
	// These test cases provide both origin, s, and the expected result.
	// If you find a bug in the while, this is probably the easiest place
	// to add it as a test case.
	var tests_wild = []struct{ e1, e2, expected string }{
		{"mathoverflow.net.", ".", "mathoverflow.net"},
		{"mathoverflow.net", ".", "mathoverflow.net"},
		{"", ".", "@"},
		{"@", ".", "@"},
	}
	for i, test := range tests_wild {
		actual := TrimDomainName(test.e1, test.e2)
		if test.expected != actual {
			t.Errorf("%d TrimDomainName(%#v, %#v): expected (%v) got (%v)\n", i, test.e1, test.e2, test.expected, actual)
		}
	}

}
