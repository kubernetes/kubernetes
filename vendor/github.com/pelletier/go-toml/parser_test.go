package toml

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
)

func assertSubTree(t *testing.T, path []string, tree *TomlTree, err error, ref map[string]interface{}) {
	if err != nil {
		t.Error("Non-nil error:", err.Error())
		return
	}
	for k, v := range ref {
		nextPath := append(path, k)
		t.Log("asserting path", nextPath)
		// NOTE: directly access key instead of resolve by path
		// NOTE: see TestSpecialKV
		switch node := tree.GetPath([]string{k}).(type) {
		case []*TomlTree:
			t.Log("\tcomparing key", nextPath, "by array iteration")
			for idx, item := range node {
				assertSubTree(t, nextPath, item, err, v.([]map[string]interface{})[idx])
			}
		case *TomlTree:
			t.Log("\tcomparing key", nextPath, "by subtree assestion")
			assertSubTree(t, nextPath, node, err, v.(map[string]interface{}))
		default:
			t.Log("\tcomparing key", nextPath, "by string representation because it's of type", reflect.TypeOf(node))
			if fmt.Sprintf("%v", node) != fmt.Sprintf("%v", v) {
				t.Errorf("was expecting %v at %v but got %v", v, k, node)
			}
		}
	}
}

func assertTree(t *testing.T, tree *TomlTree, err error, ref map[string]interface{}) {
	t.Log("Asserting tree:\n", spew.Sdump(tree))
	assertSubTree(t, []string{}, tree, err, ref)
	t.Log("Finished tree assertion.")
}

func TestCreateSubTree(t *testing.T) {
	tree := newTomlTree()
	tree.createSubTree([]string{"a", "b", "c"}, Position{})
	tree.Set("a.b.c", 42)
	if tree.Get("a.b.c") != 42 {
		t.Fail()
	}
}

func TestSimpleKV(t *testing.T) {
	tree, err := Load("a = 42")
	assertTree(t, tree, err, map[string]interface{}{
		"a": int64(42),
	})

	tree, _ = Load("a = 42\nb = 21")
	assertTree(t, tree, err, map[string]interface{}{
		"a": int64(42),
		"b": int64(21),
	})
}

func TestNumberInKey(t *testing.T) {
	tree, err := Load("hello2 = 42")
	assertTree(t, tree, err, map[string]interface{}{
		"hello2": int64(42),
	})
}

func TestSimpleNumbers(t *testing.T) {
	tree, err := Load("a = +42\nb = -21\nc = +4.2\nd = -2.1")
	assertTree(t, tree, err, map[string]interface{}{
		"a": int64(42),
		"b": int64(-21),
		"c": float64(4.2),
		"d": float64(-2.1),
	})
}

func TestNumbersWithUnderscores(t *testing.T) {
	tree, err := Load("a = 1_000")
	assertTree(t, tree, err, map[string]interface{}{
		"a": int64(1000),
	})

	tree, err = Load("a = 5_349_221")
	assertTree(t, tree, err, map[string]interface{}{
		"a": int64(5349221),
	})

	tree, err = Load("a = 1_2_3_4_5")
	assertTree(t, tree, err, map[string]interface{}{
		"a": int64(12345),
	})

	tree, err = Load("flt8 = 9_224_617.445_991_228_313")
	assertTree(t, tree, err, map[string]interface{}{
		"flt8": float64(9224617.445991228313),
	})

	tree, err = Load("flt9 = 1e1_00")
	assertTree(t, tree, err, map[string]interface{}{
		"flt9": float64(1e100),
	})
}

func TestFloatsWithExponents(t *testing.T) {
	tree, err := Load("a = 5e+22\nb = 5E+22\nc = -5e+22\nd = -5e-22\ne = 6.626e-34")
	assertTree(t, tree, err, map[string]interface{}{
		"a": float64(5e+22),
		"b": float64(5E+22),
		"c": float64(-5e+22),
		"d": float64(-5e-22),
		"e": float64(6.626e-34),
	})
}

func TestSimpleDate(t *testing.T) {
	tree, err := Load("a = 1979-05-27T07:32:00Z")
	assertTree(t, tree, err, map[string]interface{}{
		"a": time.Date(1979, time.May, 27, 7, 32, 0, 0, time.UTC),
	})
}

func TestDateOffset(t *testing.T) {
	tree, err := Load("a = 1979-05-27T00:32:00-07:00")
	assertTree(t, tree, err, map[string]interface{}{
		"a": time.Date(1979, time.May, 27, 0, 32, 0, 0, time.FixedZone("", -7*60*60)),
	})
}

func TestDateNano(t *testing.T) {
	tree, err := Load("a = 1979-05-27T00:32:00.999999999-07:00")
	assertTree(t, tree, err, map[string]interface{}{
		"a": time.Date(1979, time.May, 27, 0, 32, 0, 999999999, time.FixedZone("", -7*60*60)),
	})
}

func TestSimpleString(t *testing.T) {
	tree, err := Load("a = \"hello world\"")
	assertTree(t, tree, err, map[string]interface{}{
		"a": "hello world",
	})
}

func TestSpaceKey(t *testing.T) {
	tree, err := Load("\"a b\" = \"hello world\"")
	assertTree(t, tree, err, map[string]interface{}{
		"a b": "hello world",
	})
}

func TestStringEscapables(t *testing.T) {
	tree, err := Load("a = \"a \\n b\"")
	assertTree(t, tree, err, map[string]interface{}{
		"a": "a \n b",
	})

	tree, err = Load("a = \"a \\t b\"")
	assertTree(t, tree, err, map[string]interface{}{
		"a": "a \t b",
	})

	tree, err = Load("a = \"a \\r b\"")
	assertTree(t, tree, err, map[string]interface{}{
		"a": "a \r b",
	})

	tree, err = Load("a = \"a \\\\ b\"")
	assertTree(t, tree, err, map[string]interface{}{
		"a": "a \\ b",
	})
}

func TestBools(t *testing.T) {
	tree, err := Load("a = true\nb = false")
	assertTree(t, tree, err, map[string]interface{}{
		"a": true,
		"b": false,
	})
}

func TestNestedKeys(t *testing.T) {
	tree, err := Load("[a.b.c]\nd = 42")
	assertTree(t, tree, err, map[string]interface{}{
		"a": map[string]interface{}{
			"b": map[string]interface{}{
				"c": map[string]interface{}{
					"d": int64(42),
				},
			},
		},
	})
}

func TestNestedQuotedUnicodeKeys(t *testing.T) {
	tree, err := Load("[ j . \"ʞ\" . l ]\nd = 42")
	assertTree(t, tree, err, map[string]interface{}{
		"j": map[string]interface{}{
			"ʞ": map[string]interface{}{
				"l": map[string]interface{}{
					"d": int64(42),
				},
			},
		},
	})

	tree, err = Load("[ g . h . i ]\nd = 42")
	assertTree(t, tree, err, map[string]interface{}{
		"g": map[string]interface{}{
			"h": map[string]interface{}{
				"i": map[string]interface{}{
					"d": int64(42),
				},
			},
		},
	})

	tree, err = Load("[ d.e.f ]\nk = 42")
	assertTree(t, tree, err, map[string]interface{}{
		"d": map[string]interface{}{
			"e": map[string]interface{}{
				"f": map[string]interface{}{
					"k": int64(42),
				},
			},
		},
	})
}

func TestArrayOne(t *testing.T) {
	tree, err := Load("a = [1]")
	assertTree(t, tree, err, map[string]interface{}{
		"a": []int64{int64(1)},
	})
}

func TestArrayZero(t *testing.T) {
	tree, err := Load("a = []")
	assertTree(t, tree, err, map[string]interface{}{
		"a": []interface{}{},
	})
}

func TestArraySimple(t *testing.T) {
	tree, err := Load("a = [42, 21, 10]")
	assertTree(t, tree, err, map[string]interface{}{
		"a": []int64{int64(42), int64(21), int64(10)},
	})

	tree, _ = Load("a = [42, 21, 10,]")
	assertTree(t, tree, err, map[string]interface{}{
		"a": []int64{int64(42), int64(21), int64(10)},
	})
}

func TestArrayMultiline(t *testing.T) {
	tree, err := Load("a = [42,\n21, 10,]")
	assertTree(t, tree, err, map[string]interface{}{
		"a": []int64{int64(42), int64(21), int64(10)},
	})
}

func TestArrayNested(t *testing.T) {
	tree, err := Load("a = [[42, 21], [10]]")
	assertTree(t, tree, err, map[string]interface{}{
		"a": [][]int64{[]int64{int64(42), int64(21)}, []int64{int64(10)}},
	})
}

func TestNestedEmptyArrays(t *testing.T) {
	tree, err := Load("a = [[[]]]")
	assertTree(t, tree, err, map[string]interface{}{
		"a": [][][]interface{}{[][]interface{}{[]interface{}{}}},
	})
}

func TestArrayMixedTypes(t *testing.T) {
	_, err := Load("a = [42, 16.0]")
	if err.Error() != "(1, 10): mixed types in array" {
		t.Error("Bad error message:", err.Error())
	}

	_, err = Load("a = [42, \"hello\"]")
	if err.Error() != "(1, 11): mixed types in array" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestArrayNestedStrings(t *testing.T) {
	tree, err := Load("data = [ [\"gamma\", \"delta\"], [\"Foo\"] ]")
	assertTree(t, tree, err, map[string]interface{}{
		"data": [][]string{[]string{"gamma", "delta"}, []string{"Foo"}},
	})
}

func TestParseUnknownRvalue(t *testing.T) {
	_, err := Load("a = !bssss")
	if err == nil {
		t.Error("Expecting a parse error")
	}

	_, err = Load("a = /b")
	if err == nil {
		t.Error("Expecting a parse error")
	}
}

func TestMissingValue(t *testing.T) {
	_, err := Load("a = ")
	if err.Error() != "(1, 5): expecting a value" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestUnterminatedArray(t *testing.T) {
	_, err := Load("a = [1,")
	if err.Error() != "(1, 8): unterminated array" {
		t.Error("Bad error message:", err.Error())
	}

	_, err = Load("a = [1")
	if err.Error() != "(1, 7): unterminated array" {
		t.Error("Bad error message:", err.Error())
	}

	_, err = Load("a = [1 2")
	if err.Error() != "(1, 8): missing comma" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestNewlinesInArrays(t *testing.T) {
	tree, err := Load("a = [1,\n2,\n3]")
	assertTree(t, tree, err, map[string]interface{}{
		"a": []int64{int64(1), int64(2), int64(3)},
	})
}

func TestArrayWithExtraComma(t *testing.T) {
	tree, err := Load("a = [1,\n2,\n3,\n]")
	assertTree(t, tree, err, map[string]interface{}{
		"a": []int64{int64(1), int64(2), int64(3)},
	})
}

func TestArrayWithExtraCommaComment(t *testing.T) {
	tree, err := Load("a = [1, # wow\n2, # such items\n3, # so array\n]")
	assertTree(t, tree, err, map[string]interface{}{
		"a": []int64{int64(1), int64(2), int64(3)},
	})
}

func TestSimpleInlineGroup(t *testing.T) {
	tree, err := Load("key = {a = 42}")
	assertTree(t, tree, err, map[string]interface{}{
		"key": map[string]interface{}{
			"a": int64(42),
		},
	})
}

func TestDoubleInlineGroup(t *testing.T) {
	tree, err := Load("key = {a = 42, b = \"foo\"}")
	assertTree(t, tree, err, map[string]interface{}{
		"key": map[string]interface{}{
			"a": int64(42),
			"b": "foo",
		},
	})
}

func TestExampleInlineGroup(t *testing.T) {
	tree, err := Load(`name = { first = "Tom", last = "Preston-Werner" }
point = { x = 1, y = 2 }`)
	assertTree(t, tree, err, map[string]interface{}{
		"name": map[string]interface{}{
			"first": "Tom",
			"last":  "Preston-Werner",
		},
		"point": map[string]interface{}{
			"x": int64(1),
			"y": int64(2),
		},
	})
}

func TestExampleInlineGroupInArray(t *testing.T) {
	tree, err := Load(`points = [{ x = 1, y = 2 }]`)
	assertTree(t, tree, err, map[string]interface{}{
		"points": []map[string]interface{}{
			map[string]interface{}{
				"x": int64(1),
				"y": int64(2),
			},
		},
	})
}

func TestInlineTableUnterminated(t *testing.T) {
	_, err := Load("foo = {")
	if err.Error() != "(1, 8): unterminated inline table" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestInlineTableCommaExpected(t *testing.T) {
	_, err := Load("foo = {hello = 53 test = foo}")
	if err.Error() != "(1, 19): comma expected between fields in inline table" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestInlineTableCommaStart(t *testing.T) {
	_, err := Load("foo = {, hello = 53}")
	if err.Error() != "(1, 8): inline table cannot start with a comma" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestInlineTableDoubleComma(t *testing.T) {
	_, err := Load("foo = {hello = 53,, foo = 17}")
	if err.Error() != "(1, 19): need field between two commas in inline table" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestDuplicateGroups(t *testing.T) {
	_, err := Load("[foo]\na=2\n[foo]b=3")
	if err.Error() != "(3, 2): duplicated tables" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestDuplicateKeys(t *testing.T) {
	_, err := Load("foo = 2\nfoo = 3")
	if err.Error() != "(2, 1): The following key was defined twice: foo" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestEmptyIntermediateTable(t *testing.T) {
	_, err := Load("[foo..bar]")
	if err.Error() != "(1, 2): empty intermediate table" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestImplicitDeclarationBefore(t *testing.T) {
	tree, err := Load("[a.b.c]\nanswer = 42\n[a]\nbetter = 43")
	assertTree(t, tree, err, map[string]interface{}{
		"a": map[string]interface{}{
			"b": map[string]interface{}{
				"c": map[string]interface{}{
					"answer": int64(42),
				},
			},
			"better": int64(43),
		},
	})
}

func TestFloatsWithoutLeadingZeros(t *testing.T) {
	_, err := Load("a = .42")
	if err.Error() != "(1, 5): cannot start float with a dot" {
		t.Error("Bad error message:", err.Error())
	}

	_, err = Load("a = -.42")
	if err.Error() != "(1, 5): cannot start float with a dot" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestMissingFile(t *testing.T) {
	_, err := LoadFile("foo.toml")
	if err.Error() != "open foo.toml: no such file or directory" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestParseFile(t *testing.T) {
	tree, err := LoadFile("example.toml")

	assertTree(t, tree, err, map[string]interface{}{
		"title": "TOML Example",
		"owner": map[string]interface{}{
			"name":         "Tom Preston-Werner",
			"organization": "GitHub",
			"bio":          "GitHub Cofounder & CEO\nLikes tater tots and beer.",
			"dob":          time.Date(1979, time.May, 27, 7, 32, 0, 0, time.UTC),
		},
		"database": map[string]interface{}{
			"server":         "192.168.1.1",
			"ports":          []int64{8001, 8001, 8002},
			"connection_max": 5000,
			"enabled":        true,
		},
		"servers": map[string]interface{}{
			"alpha": map[string]interface{}{
				"ip": "10.0.0.1",
				"dc": "eqdc10",
			},
			"beta": map[string]interface{}{
				"ip": "10.0.0.2",
				"dc": "eqdc10",
			},
		},
		"clients": map[string]interface{}{
			"data": []interface{}{
				[]string{"gamma", "delta"},
				[]int64{1, 2},
			},
		},
	})
}

func TestParseFileCRLF(t *testing.T) {
	tree, err := LoadFile("example-crlf.toml")

	assertTree(t, tree, err, map[string]interface{}{
		"title": "TOML Example",
		"owner": map[string]interface{}{
			"name":         "Tom Preston-Werner",
			"organization": "GitHub",
			"bio":          "GitHub Cofounder & CEO\nLikes tater tots and beer.",
			"dob":          time.Date(1979, time.May, 27, 7, 32, 0, 0, time.UTC),
		},
		"database": map[string]interface{}{
			"server":         "192.168.1.1",
			"ports":          []int64{8001, 8001, 8002},
			"connection_max": 5000,
			"enabled":        true,
		},
		"servers": map[string]interface{}{
			"alpha": map[string]interface{}{
				"ip": "10.0.0.1",
				"dc": "eqdc10",
			},
			"beta": map[string]interface{}{
				"ip": "10.0.0.2",
				"dc": "eqdc10",
			},
		},
		"clients": map[string]interface{}{
			"data": []interface{}{
				[]string{"gamma", "delta"},
				[]int64{1, 2},
			},
		},
	})
}

func TestParseKeyGroupArray(t *testing.T) {
	tree, err := Load("[[foo.bar]] a = 42\n[[foo.bar]] a = 69")
	assertTree(t, tree, err, map[string]interface{}{
		"foo": map[string]interface{}{
			"bar": []map[string]interface{}{
				{"a": int64(42)},
				{"a": int64(69)},
			},
		},
	})
}

func TestParseKeyGroupArrayUnfinished(t *testing.T) {
	_, err := Load("[[foo.bar]\na = 42")
	if err.Error() != "(1, 10): was expecting token [[, but got unclosed key group array instead" {
		t.Error("Bad error message:", err.Error())
	}

	_, err = Load("[[foo.[bar]\na = 42")
	if err.Error() != "(1, 3): unexpected token group name cannot contain ']', was expecting a key group array" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestParseKeyGroupArrayQueryExample(t *testing.T) {
	tree, err := Load(`
      [[book]]
      title = "The Stand"
      author = "Stephen King"
      [[book]]
      title = "For Whom the Bell Tolls"
      author = "Ernest Hemmingway"
      [[book]]
      title = "Neuromancer"
      author = "William Gibson"
    `)

	assertTree(t, tree, err, map[string]interface{}{
		"book": []map[string]interface{}{
			{"title": "The Stand", "author": "Stephen King"},
			{"title": "For Whom the Bell Tolls", "author": "Ernest Hemmingway"},
			{"title": "Neuromancer", "author": "William Gibson"},
		},
	})
}

func TestParseKeyGroupArraySpec(t *testing.T) {
	tree, err := Load("[[fruit]]\n name=\"apple\"\n [fruit.physical]\n color=\"red\"\n shape=\"round\"\n [[fruit]]\n name=\"banana\"")
	assertTree(t, tree, err, map[string]interface{}{
		"fruit": []map[string]interface{}{
			{"name": "apple", "physical": map[string]interface{}{"color": "red", "shape": "round"}},
			{"name": "banana"},
		},
	})
}

func TestToTomlValue(t *testing.T) {
	for idx, item := range []struct {
		Value  interface{}
		Expect string
	}{
		{int64(12345), "12345"},
		{float64(123.45), "123.45"},
		{bool(true), "true"},
		{"hello world", "\"hello world\""},
		{"\b\t\n\f\r\"\\", "\"\\b\\t\\n\\f\\r\\\"\\\\\""},
		{"\x05", "\"\\u0005\""},
		{time.Date(1979, time.May, 27, 7, 32, 0, 0, time.UTC),
			"1979-05-27T07:32:00Z"},
		{[]interface{}{"gamma", "delta"},
			"[\n  \"gamma\",\n  \"delta\",\n]"},
	} {
		result := toTomlValue(item.Value, 0)
		if result != item.Expect {
			t.Errorf("Test %d - got '%s', expected '%s'", idx, result, item.Expect)
		}
	}
}

func TestToString(t *testing.T) {
	tree, err := Load("[foo]\n\n[[foo.bar]]\na = 42\n\n[[foo.bar]]\na = 69\n")
	if err != nil {
		t.Errorf("Test failed to parse: %v", err)
		return
	}
	result := tree.ToString()
	expected := "\n[foo]\n\n  [[foo.bar]]\n    a = 42\n\n  [[foo.bar]]\n    a = 69\n"
	if result != expected {
		t.Errorf("Expected got '%s', expected '%s'", result, expected)
	}
}

func assertPosition(t *testing.T, text string, ref map[string]Position) {
	tree, err := Load(text)
	if err != nil {
		t.Errorf("Error loading document text: `%v`", text)
		t.Errorf("Error: %v", err)
	}
	for path, pos := range ref {
		testPos := tree.GetPosition(path)
		if testPos.Invalid() {
			t.Errorf("Failed to query tree path or path has invalid position: %s", path)
		} else if pos != testPos {
			t.Errorf("Expected position %v, got %v instead", pos, testPos)
		}
	}
}

func TestDocumentPositions(t *testing.T) {
	assertPosition(t,
		"[foo]\nbar=42\nbaz=69",
		map[string]Position{
			"":        Position{1, 1},
			"foo":     Position{1, 1},
			"foo.bar": Position{2, 1},
			"foo.baz": Position{3, 1},
		})
}

func TestDocumentPositionsWithSpaces(t *testing.T) {
	assertPosition(t,
		"  [foo]\n  bar=42\n  baz=69",
		map[string]Position{
			"":        Position{1, 1},
			"foo":     Position{1, 3},
			"foo.bar": Position{2, 3},
			"foo.baz": Position{3, 3},
		})
}

func TestDocumentPositionsWithGroupArray(t *testing.T) {
	assertPosition(t,
		"[[foo]]\nbar=42\nbaz=69",
		map[string]Position{
			"":        Position{1, 1},
			"foo":     Position{1, 1},
			"foo.bar": Position{2, 1},
			"foo.baz": Position{3, 1},
		})
}

func TestNestedTreePosition(t *testing.T) {
	assertPosition(t,
		"[foo.bar]\na=42\nb=69",
		map[string]Position{
			"":          Position{1, 1},
			"foo":       Position{1, 1},
			"foo.bar":   Position{1, 1},
			"foo.bar.a": Position{2, 1},
			"foo.bar.b": Position{3, 1},
		})
}

func TestInvalidGroupArray(t *testing.T) {
	_, err := Load("[key#group]\nanswer = 42")
	if err == nil {
		t.Error("Should error")
	}

	_, err = Load("[foo.[bar]\na = 42")
	if err.Error() != "(1, 2): unexpected token group name cannot contain ']', was expecting a key group" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestDoubleEqual(t *testing.T) {
	_, err := Load("foo= = 2")
	if err.Error() != "(1, 6): cannot have multiple equals for the same key" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestGroupArrayReassign(t *testing.T) {
	_, err := Load("[hello]\n[[hello]]")
	if err.Error() != "(2, 3): key \"hello\" is already assigned and not of type group array" {
		t.Error("Bad error message:", err.Error())
	}
}

func TestInvalidFloatParsing(t *testing.T) {
	_, err := Load("a=1e_2")
	if err.Error() != "(1, 3): invalid use of _ in number" {
		t.Error("Bad error message:", err.Error())
	}

	_, err = Load("a=1e2_")
	if err.Error() != "(1, 3): invalid use of _ in number" {
		t.Error("Bad error message:", err.Error())
	}

	_, err = Load("a=1__2")
	if err.Error() != "(1, 3): invalid use of _ in number" {
		t.Error("Bad error message:", err.Error())
	}

	_, err = Load("a=_1_2")
	if err.Error() != "(1, 3): cannot start number with underscore" {
		t.Error("Bad error message:", err.Error())
	}
}
