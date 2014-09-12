package ini

import (
	"reflect"
	"strings"
	"testing"
)

func TestLoad(t *testing.T) {
	src := `
  # Comments are ignored

  herp = derp

  [foo]
  hello=world
  whitespace should   =   not matter   
  ; sneaky semicolon-style comment
  multiple = equals = signs

  [bar]
  this = that`

	file, err := Load(strings.NewReader(src))
	if err != nil {
		t.Fatal(err)
	}
	check := func(section, key, expect string) {
		if value, _ := file.Get(section, key); value != expect {
			t.Errorf("Get(%q, %q): expected %q, got %q", section, key, expect, value)
		}
	}

	check("", "herp", "derp")
	check("foo", "hello", "world")
	check("foo", "whitespace should", "not matter")
	check("foo", "multiple", "equals = signs")
	check("bar", "this", "that")
}

func TestSyntaxError(t *testing.T) {
	src := `
  # Line 2
  [foo]
  bar = baz
  # Here's an error on line 6:
  wut?
  herp = derp`
	_, err := Load(strings.NewReader(src))
	t.Logf("%T: %v", err, err)
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	syntaxErr, ok := err.(ErrSyntax)
	if !ok {
		t.Fatal("expected an error of type ErrSyntax")
	}
	if syntaxErr.Line != 6 {
		t.Fatal("incorrect line number")
	}
	if syntaxErr.Source != "wut?" {
		t.Fatal("incorrect source")
	}
}

func TestDefinedSectionBehaviour(t *testing.T) {
	check := func(src string, expect File) {
		file, err := Load(strings.NewReader(src))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(file, expect) {
			t.Errorf("expected %v, got %v", expect, file)
		}
	}
	// No sections for an empty file
	check("", File{})
	// Default section only if there are actually values for it
	check("foo=bar", File{"": {"foo": "bar"}})
	// User-defined sections should always be present, even if empty
	check("[a]\n[b]\nfoo=bar", File{
		"a": {},
		"b": {"foo": "bar"},
	})
	check("foo=bar\n[a]\nthis=that", File{
		"":  {"foo": "bar"},
		"a": {"this": "that"},
	})
}
