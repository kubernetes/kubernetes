package parser

import (
	"fmt"
	"io/ioutil"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"github.com/hashicorp/hcl/hcl/ast"
	"github.com/hashicorp/hcl/hcl/token"
)

func TestType(t *testing.T) {
	var literals = []struct {
		typ token.Type
		src string
	}{
		{token.STRING, `"foo": "bar"`},
		{token.NUMBER, `"foo": 123`},
		{token.FLOAT, `"foo": 123.12`},
		{token.FLOAT, `"foo": -123.12`},
		{token.BOOL, `"foo": true`},
		{token.STRING, `"foo": null`},
	}

	for _, l := range literals {
		t.Logf("Testing: %s", l.src)

		p := newParser([]byte(l.src))
		item, err := p.objectItem()
		if err != nil {
			t.Error(err)
		}

		lit, ok := item.Val.(*ast.LiteralType)
		if !ok {
			t.Errorf("node should be of type LiteralType, got: %T", item.Val)
		}

		if lit.Token.Type != l.typ {
			t.Errorf("want: %s, got: %s", l.typ, lit.Token.Type)
		}
	}
}

func TestListType(t *testing.T) {
	var literals = []struct {
		src    string
		tokens []token.Type
	}{
		{
			`"foo": ["123", 123]`,
			[]token.Type{token.STRING, token.NUMBER},
		},
		{
			`"foo": [123, "123",]`,
			[]token.Type{token.NUMBER, token.STRING},
		},
		{
			`"foo": []`,
			[]token.Type{},
		},
		{
			`"foo": ["123", 123]`,
			[]token.Type{token.STRING, token.NUMBER},
		},
		{
			`"foo": ["123", {}]`,
			[]token.Type{token.STRING, token.LBRACE},
		},
	}

	for _, l := range literals {
		t.Logf("Testing: %s", l.src)

		p := newParser([]byte(l.src))
		item, err := p.objectItem()
		if err != nil {
			t.Error(err)
		}

		list, ok := item.Val.(*ast.ListType)
		if !ok {
			t.Errorf("node should be of type LiteralType, got: %T", item.Val)
		}

		tokens := []token.Type{}
		for _, li := range list.List {
			switch v := li.(type) {
			case *ast.LiteralType:
				tokens = append(tokens, v.Token.Type)
			case *ast.ObjectType:
				tokens = append(tokens, token.LBRACE)
			}
		}

		equals(t, l.tokens, tokens)
	}
}

func TestObjectType(t *testing.T) {
	var literals = []struct {
		src      string
		nodeType []ast.Node
		itemLen  int
	}{
		{
			`"foo": {}`,
			nil,
			0,
		},
		{
			`"foo": {
				"bar": "fatih"
			 }`,
			[]ast.Node{&ast.LiteralType{}},
			1,
		},
		{
			`"foo": {
				"bar": "fatih",
				"baz": ["arslan"]
			 }`,
			[]ast.Node{
				&ast.LiteralType{},
				&ast.ListType{},
			},
			2,
		},
		{
			`"foo": {
				"bar": {}
			 }`,
			[]ast.Node{
				&ast.ObjectType{},
			},
			1,
		},
		{
			`"foo": {
				"bar": {},
				"foo": true
			 }`,
			[]ast.Node{
				&ast.ObjectType{},
				&ast.LiteralType{},
			},
			2,
		},
	}

	for _, l := range literals {
		t.Logf("Testing:\n%s\n", l.src)

		p := newParser([]byte(l.src))
		// p.enableTrace = true
		item, err := p.objectItem()
		if err != nil {
			t.Error(err)
		}

		// we know that the ObjectKey name is foo for all cases, what matters
		// is the object
		obj, ok := item.Val.(*ast.ObjectType)
		if !ok {
			t.Errorf("node should be of type LiteralType, got: %T", item.Val)
		}

		// check if the total length of items are correct
		equals(t, l.itemLen, len(obj.List.Items))

		// check if the types are correct
		for i, item := range obj.List.Items {
			equals(t, reflect.TypeOf(l.nodeType[i]), reflect.TypeOf(item.Val))
		}
	}
}

func TestFlattenObjects(t *testing.T) {
	var literals = []struct {
		src      string
		nodeType []ast.Node
		itemLen  int
	}{
		{
			`{
				"foo": [
					{
						"foo": "svh",
						"bar": "fatih"
					}
				]
			}`,
			[]ast.Node{
				&ast.ObjectType{},
				&ast.LiteralType{},
				&ast.LiteralType{},
			},
			3,
		},
		{
			`{
				"variable": {
					"foo": {}
				}
			}`,
			[]ast.Node{
				&ast.ObjectType{},
			},
			1,
		},
	}

	for _, l := range literals {
		t.Logf("Testing:\n%s\n", l.src)

		f, err := Parse([]byte(l.src))
		if err != nil {
			t.Error(err)
		}

		// the first object is always an ObjectList so just assert that one
		// so we can use it as such
		obj, ok := f.Node.(*ast.ObjectList)
		if !ok {
			t.Errorf("node should be *ast.ObjectList, got: %T", f.Node)
		}

		// check if the types are correct
		var i int
		for _, item := range obj.Items {
			equals(t, reflect.TypeOf(l.nodeType[i]), reflect.TypeOf(item.Val))
			i++

			if obj, ok := item.Val.(*ast.ObjectType); ok {
				for _, item := range obj.List.Items {
					equals(t, reflect.TypeOf(l.nodeType[i]), reflect.TypeOf(item.Val))
					i++
				}
			}
		}

		// check if the number of items is correct
		equals(t, l.itemLen, i)

	}
}

func TestObjectKey(t *testing.T) {
	keys := []struct {
		exp []token.Type
		src string
	}{
		{[]token.Type{token.STRING}, `"foo": {}`},
	}

	for _, k := range keys {
		p := newParser([]byte(k.src))
		keys, err := p.objectKey()
		if err != nil {
			t.Fatal(err)
		}

		tokens := []token.Type{}
		for _, o := range keys {
			tokens = append(tokens, o.Token.Type)
		}

		equals(t, k.exp, tokens)
	}

	errKeys := []struct {
		src string
	}{
		{`foo 12 {}`},
		{`foo bar = {}`},
		{`foo []`},
		{`12 {}`},
	}

	for _, k := range errKeys {
		p := newParser([]byte(k.src))
		_, err := p.objectKey()
		if err == nil {
			t.Errorf("case '%s' should give an error", k.src)
		}
	}
}

// Official HCL tests
func TestParse(t *testing.T) {
	cases := []struct {
		Name string
		Err  bool
	}{
		{
			"array.json",
			false,
		},
		{
			"basic.json",
			false,
		},
		{
			"object.json",
			false,
		},
		{
			"types.json",
			false,
		},
		{
			"bad_input_128.json",
			true,
		},
	}

	const fixtureDir = "./test-fixtures"

	for _, tc := range cases {
		d, err := ioutil.ReadFile(filepath.Join(fixtureDir, tc.Name))
		if err != nil {
			t.Fatalf("err: %s", err)
		}

		_, err = Parse(d)
		if (err != nil) != tc.Err {
			t.Fatalf("Input: %s\n\nError: %s", tc.Name, err)
		}
	}
}

func TestParse_inline(t *testing.T) {
	cases := []struct {
		Value string
		Err   bool
	}{
		{"{:{", true},
	}

	for _, tc := range cases {
		_, err := Parse([]byte(tc.Value))
		if (err != nil) != tc.Err {
			t.Fatalf("Input: %q\n\nError: %s", tc.Value, err)
		}
	}
}

// equals fails the test if exp is not equal to act.
func equals(tb testing.TB, exp, act interface{}) {
	if !reflect.DeepEqual(exp, act) {
		_, file, line, _ := runtime.Caller(1)
		fmt.Printf("\033[31m%s:%d:\n\n\texp: %#v\n\n\tgot: %#v\033[39m\n\n", filepath.Base(file), line, exp, act)
		tb.FailNow()
	}
}
