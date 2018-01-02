package parser

import (
	"fmt"
	"io/ioutil"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/hashicorp/hcl/hcl/ast"
	"github.com/hashicorp/hcl/hcl/token"
)

func TestType(t *testing.T) {
	var literals = []struct {
		typ token.Type
		src string
	}{
		{token.STRING, `foo = "foo"`},
		{token.NUMBER, `foo = 123`},
		{token.NUMBER, `foo = -29`},
		{token.FLOAT, `foo = 123.12`},
		{token.FLOAT, `foo = -123.12`},
		{token.BOOL, `foo = true`},
		{token.HEREDOC, "foo = <<EOF\nHello\nWorld\nEOF"},
	}

	for _, l := range literals {
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
			`foo = ["123", 123]`,
			[]token.Type{token.STRING, token.NUMBER},
		},
		{
			`foo = [123, "123",]`,
			[]token.Type{token.NUMBER, token.STRING},
		},
		{
			`foo = []`,
			[]token.Type{},
		},
		{
			`foo = ["123", 123]`,
			[]token.Type{token.STRING, token.NUMBER},
		},
		{
			`foo = [1,
"string",
<<EOF
heredoc contents
EOF
]`,
			[]token.Type{token.NUMBER, token.STRING, token.HEREDOC},
		},
	}

	for _, l := range literals {
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
			if tp, ok := li.(*ast.LiteralType); ok {
				tokens = append(tokens, tp.Token.Type)
			}
		}

		equals(t, l.tokens, tokens)
	}
}

func TestListOfMaps(t *testing.T) {
	src := `foo = [
    {key = "bar"},
    {key = "baz", key2 = "qux"},
  ]`
	p := newParser([]byte(src))

	file, err := p.Parse()
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Here we make all sorts of assumptions about the input structure w/ type
	// assertions. The intent is only for this to be a "smoke test" ensuring
	// parsing actually performed its duty - giving this test something a bit
	// more robust than _just_ "no error occurred".
	expected := []string{`"bar"`, `"baz"`, `"qux"`}
	actual := make([]string, 0, 3)
	ol := file.Node.(*ast.ObjectList)
	objItem := ol.Items[0]
	list := objItem.Val.(*ast.ListType)
	for _, node := range list.List {
		obj := node.(*ast.ObjectType)
		for _, item := range obj.List.Items {
			val := item.Val.(*ast.LiteralType)
			actual = append(actual, val.Token.Text)
		}

	}
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("Expected: %#v, got %#v", expected, actual)
	}
}

func TestListOfMaps_requiresComma(t *testing.T) {
	src := `foo = [
    {key = "bar"}
    {key = "baz"}
  ]`
	p := newParser([]byte(src))

	_, err := p.Parse()
	if err == nil {
		t.Fatalf("Expected error, got none!")
	}

	expected := "error parsing list, expected comma or list end"
	if !strings.Contains(err.Error(), expected) {
		t.Fatalf("Expected err:\n  %s\nTo contain:\n  %s\n", err, expected)
	}
}

func TestObjectType(t *testing.T) {
	var literals = []struct {
		src      string
		nodeType []ast.Node
		itemLen  int
	}{
		{
			`foo = {}`,
			nil,
			0,
		},
		{
			`foo = {
				bar = "fatih"
			 }`,
			[]ast.Node{&ast.LiteralType{}},
			1,
		},
		{
			`foo = {
				bar = "fatih"
				baz = ["arslan"]
			 }`,
			[]ast.Node{
				&ast.LiteralType{},
				&ast.ListType{},
			},
			2,
		},
		{
			`foo = {
				bar {}
			 }`,
			[]ast.Node{
				&ast.ObjectType{},
			},
			1,
		},
		{
			`foo {
				bar {}
				foo = true
			 }`,
			[]ast.Node{
				&ast.ObjectType{},
				&ast.LiteralType{},
			},
			2,
		},
	}

	for _, l := range literals {
		p := newParser([]byte(l.src))
		// p.enableTrace = true
		item, err := p.objectItem()
		if err != nil {
			t.Error(err)
			continue
		}

		// we know that the ObjectKey name is foo for all cases, what matters
		// is the object
		obj, ok := item.Val.(*ast.ObjectType)
		if !ok {
			t.Errorf("node should be of type LiteralType, got: %T", item.Val)
			continue
		}

		// check if the total length of items are correct
		equals(t, l.itemLen, len(obj.List.Items))

		// check if the types are correct
		for i, item := range obj.List.Items {
			equals(t, reflect.TypeOf(l.nodeType[i]), reflect.TypeOf(item.Val))
		}
	}
}

func TestObjectKey(t *testing.T) {
	keys := []struct {
		exp []token.Type
		src string
	}{
		{[]token.Type{token.IDENT}, `foo {}`},
		{[]token.Type{token.IDENT}, `foo = {}`},
		{[]token.Type{token.IDENT}, `foo = bar`},
		{[]token.Type{token.IDENT}, `foo = 123`},
		{[]token.Type{token.IDENT}, `foo = "${var.bar}`},
		{[]token.Type{token.STRING}, `"foo" {}`},
		{[]token.Type{token.STRING}, `"foo" = {}`},
		{[]token.Type{token.STRING}, `"foo" = "${var.bar}`},
		{[]token.Type{token.IDENT, token.IDENT}, `foo bar {}`},
		{[]token.Type{token.IDENT, token.STRING}, `foo "bar" {}`},
		{[]token.Type{token.STRING, token.IDENT}, `"foo" bar {}`},
		{[]token.Type{token.IDENT, token.IDENT, token.IDENT}, `foo bar baz {}`},
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
			"assign_colon.hcl",
			true,
		},
		{
			"comment.hcl",
			false,
		},
		{
			"comment_lastline.hcl",
			false,
		},
		{
			"comment_single.hcl",
			false,
		},
		{
			"empty.hcl",
			false,
		},
		{
			"list_comma.hcl",
			false,
		},
		{
			"multiple.hcl",
			false,
		},
		{
			"object_list_comma.hcl",
			false,
		},
		{
			"structure.hcl",
			false,
		},
		{
			"structure_basic.hcl",
			false,
		},
		{
			"structure_empty.hcl",
			false,
		},
		{
			"complex.hcl",
			false,
		},
		{
			"types.hcl",
			false,
		},
		{
			"array_comment.hcl",
			false,
		},
		{
			"array_comment_2.hcl",
			true,
		},
		{
			"missing_braces.hcl",
			true,
		},
		{
			"unterminated_object.hcl",
			true,
		},
		{
			"unterminated_object_2.hcl",
			true,
		},
		{
			"key_without_value.hcl",
			true,
		},
		{
			"object_key_without_value.hcl",
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
		{"t t e{{}}", true},
		{"o{{}}", true},
		{"t t e d N{{}}", true},
		{"t t e d{{}}", true},
		{"N{}N{{}}", true},
		{"v\nN{{}}", true},
		{"v=/\n[,", true},
	}

	for _, tc := range cases {
		t.Logf("Testing: %q", tc.Value)
		_, err := Parse([]byte(tc.Value))
		if (err != nil) != tc.Err {
			t.Fatalf("Input: %q\n\nError: %s\n\nAST: %s", tc.Value, err)
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
