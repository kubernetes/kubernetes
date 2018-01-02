// +build 1.6,codegen

package api

import (
	"testing"
)

func TestNonHTMLDocGen(t *testing.T) {
	doc := "Testing 1 2 3"
	expected := "// Testing 1 2 3\n"
	doc = docstring(doc)

	if expected != doc {
		t.Errorf("Expected %s, but received %s", expected, doc)
	}
}

func TestListsHTMLDocGen(t *testing.T) {
	doc := "<ul><li>Testing 1 2 3</li> <li>FooBar</li></ul>"
	expected := "//    * Testing 1 2 3\n//    * FooBar\n"
	doc = docstring(doc)

	if expected != doc {
		t.Errorf("Expected %s, but received %s", expected, doc)
	}

	doc = "<ul> <li>Testing 1 2 3</li> <li>FooBar</li> </ul>"
	expected = "//    * Testing 1 2 3\n//    * FooBar\n"
	doc = docstring(doc)

	if expected != doc {
		t.Errorf("Expected %s, but received %s", expected, doc)
	}

	// Test leading spaces
	doc = " <ul> <li>Testing 1 2 3</li> <li>FooBar</li> </ul>"
	doc = docstring(doc)
	if expected != doc {
		t.Errorf("Expected %s, but received %s", expected, doc)
	}

	// Paragraph check
	doc = "<ul> <li> <p>Testing 1 2 3</p> </li><li> <p>FooBar</p></li></ul>"
	expected = "//    * Testing 1 2 3\n// \n//    * FooBar\n"
	doc = docstring(doc)
	if expected != doc {
		t.Errorf("Expected %s, but received %s", expected, doc)
	}
}

func TestInlineCodeHTMLDocGen(t *testing.T) {
	doc := "<ul> <li><code>Testing</code>: 1 2 3</li> <li>FooBar</li> </ul>"
	expected := "//    * Testing: 1 2 3\n//    * FooBar\n"
	doc = docstring(doc)

	if expected != doc {
		t.Errorf("Expected %s, but received %s", expected, doc)
	}
}

func TestInlineCodeInParagraphHTMLDocGen(t *testing.T) {
	doc := "<p><code>Testing</code>: 1 2 3</p>"
	expected := "// Testing: 1 2 3\n"
	doc = docstring(doc)

	if expected != doc {
		t.Errorf("Expected %s, but received %s", expected, doc)
	}
}

func TestEmptyPREInlineCodeHTMLDocGen(t *testing.T) {
	doc := "<pre><code>Testing</code></pre>"
	expected := "//    Testing\n"
	doc = docstring(doc)

	if expected != doc {
		t.Errorf("Expected %s, but received %s", expected, doc)
	}
}

func TestParagraph(t *testing.T) {
	doc := "<p>Testing 1 2 3</p>"
	expected := "// Testing 1 2 3\n"
	doc = docstring(doc)

	if expected != doc {
		t.Errorf("Expected %s, but received %s", expected, doc)
	}
}

func TestComplexListParagraphCode(t *testing.T) {
	doc := "<ul> <li><p><code>FOO</code> Bar</p></li><li><p><code>Xyz</code> ABC</p></li></ul>"
	expected := "//    * FOO Bar\n// \n//    * Xyz ABC\n"
	doc = docstring(doc)

	if expected != doc {
		t.Errorf("Expected %s, but received %s", expected, doc)
	}
}
