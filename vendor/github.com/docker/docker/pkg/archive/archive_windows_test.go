// +build windows

package archive

import (
	"os"
	"testing"
)

func TestCanonicalTarNameForPath(t *testing.T) {
	cases := []struct {
		in, expected string
		shouldFail   bool
	}{
		{"foo", "foo", false},
		{"foo/bar", "___", true}, // unix-styled windows path must fail
		{`foo\bar`, "foo/bar", false},
	}
	for _, v := range cases {
		if out, err := CanonicalTarNameForPath(v.in); err != nil && !v.shouldFail {
			t.Fatalf("cannot get canonical name for path: %s: %v", v.in, err)
		} else if v.shouldFail && err == nil {
			t.Fatalf("canonical path call should have failed with error. in=%s out=%s", v.in, out)
		} else if !v.shouldFail && out != v.expected {
			t.Fatalf("wrong canonical tar name. expected:%s got:%s", v.expected, out)
		}
	}
}

func TestCanonicalTarName(t *testing.T) {
	cases := []struct {
		in       string
		isDir    bool
		expected string
	}{
		{"foo", false, "foo"},
		{"foo", true, "foo/"},
		{`foo\bar`, false, "foo/bar"},
		{`foo\bar`, true, "foo/bar/"},
	}
	for _, v := range cases {
		if out, err := canonicalTarName(v.in, v.isDir); err != nil {
			t.Fatalf("cannot get canonical name for path: %s: %v", v.in, err)
		} else if out != v.expected {
			t.Fatalf("wrong canonical tar name. expected:%s got:%s", v.expected, out)
		}
	}
}

func TestChmodTarEntry(t *testing.T) {
	cases := []struct {
		in, expected os.FileMode
	}{
		{0000, 0111},
		{0777, 0755},
		{0644, 0755},
		{0755, 0755},
		{0444, 0555},
	}
	for _, v := range cases {
		if out := chmodTarEntry(v.in); out != v.expected {
			t.Fatalf("wrong chmod. expected:%v got:%v", v.expected, out)
		}
	}
}
