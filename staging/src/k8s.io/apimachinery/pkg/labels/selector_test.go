/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package labels

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

var (
	ignoreDetail = cmpopts.IgnoreFields(field.Error{}, "Detail")
)

func TestSelectorParse(t *testing.T) {
	testGoodStrings := []string{
		"x=a,y=b,z=c",
		"",
		"x!=a,y=b",
		"x=",
		"x= ",
		"x=,z= ",
		"x= ,z= ",
		"!x",
		"x>1",
		"x>1,z<5",
	}
	testBadStrings := []string{
		"x=a||y=b",
		"x==a==b",
		"!x=a",
		"x<a",
	}
	for _, test := range testGoodStrings {
		lq, err := Parse(test)
		if err != nil {
			t.Errorf("%v: error %v (%#v)\n", test, err, err)
		}
		if strings.Replace(test, " ", "", -1) != lq.String() {
			t.Errorf("%v restring gave: %v\n", test, lq.String())
		}
	}
	for _, test := range testBadStrings {
		_, err := Parse(test)
		if err == nil {
			t.Errorf("%v: did not get expected error\n", test)
		}
	}
}

func TestDeterministicParse(t *testing.T) {
	s1, err := Parse("x=a,a=x")
	s2, err2 := Parse("a=x,x=a")
	if err != nil || err2 != nil {
		t.Errorf("Unexpected parse error")
	}
	if s1.String() != s2.String() {
		t.Errorf("Non-deterministic parse")
	}
}

func expectMatch(t *testing.T, selector string, ls Set) {
	lq, err := Parse(selector)
	if err != nil {
		t.Errorf("Unable to parse %v as a selector\n", selector)
		return
	}
	if !lq.Matches(ls) {
		t.Errorf("Wanted %s to match '%s', but it did not.\n", selector, ls)
	}
}

func expectNoMatch(t *testing.T, selector string, ls Set) {
	lq, err := Parse(selector)
	if err != nil {
		t.Errorf("Unable to parse %v as a selector\n", selector)
		return
	}
	if lq.Matches(ls) {
		t.Errorf("Wanted '%s' to not match '%s', but it did.", selector, ls)
	}
}

func TestEverything(t *testing.T) {
	if !Everything().Matches(Set{"x": "y"}) {
		t.Errorf("Nil selector didn't match")
	}
	if !Everything().Empty() {
		t.Errorf("Everything was not empty")
	}
}

func TestSelectorMatches(t *testing.T) {
	expectMatch(t, "", Set{"x": "y"})
	expectMatch(t, "x=y", Set{"x": "y"})
	expectMatch(t, "x=y,z=w", Set{"x": "y", "z": "w"})
	expectMatch(t, "x!=y,z!=w", Set{"x": "z", "z": "a"})
	expectMatch(t, "notin=in", Set{"notin": "in"}) // in and notin in exactMatch
	expectMatch(t, "x", Set{"x": "z"})
	expectMatch(t, "!x", Set{"y": "z"})
	expectMatch(t, "x>1", Set{"x": "2"})
	expectMatch(t, "x<1", Set{"x": "0"})
	expectNoMatch(t, "x=z", Set{})
	expectNoMatch(t, "x=y", Set{"x": "z"})
	expectNoMatch(t, "x=y,z=w", Set{"x": "w", "z": "w"})
	expectNoMatch(t, "x!=y,z!=w", Set{"x": "z", "z": "w"})
	expectNoMatch(t, "x", Set{"y": "z"})
	expectNoMatch(t, "!x", Set{"x": "z"})
	expectNoMatch(t, "x>1", Set{"x": "0"})
	expectNoMatch(t, "x<1", Set{"x": "2"})

	labelset := Set{
		"foo": "bar",
		"baz": "blah",
	}
	expectMatch(t, "foo=bar", labelset)
	expectMatch(t, "baz=blah", labelset)
	expectMatch(t, "foo=bar,baz=blah", labelset)
	expectNoMatch(t, "foo=blah", labelset)
	expectNoMatch(t, "baz=bar", labelset)
	expectNoMatch(t, "foo=bar,foobar=bar,baz=blah", labelset)
}

func expectMatchDirect(t *testing.T, selector, ls Set) {
	if !SelectorFromSet(selector).Matches(ls) {
		t.Errorf("Wanted %s to match '%s', but it did not.\n", selector, ls)
	}
}

//nolint:staticcheck,unused //iccheck // U1000 currently commented out in TODO of TestSetMatches
func expectNoMatchDirect(t *testing.T, selector, ls Set) {
	if SelectorFromSet(selector).Matches(ls) {
		t.Errorf("Wanted '%s' to not match '%s', but it did.", selector, ls)
	}
}

func TestSetMatches(t *testing.T) {
	labelset := Set{
		"foo": "bar",
		"baz": "blah",
	}
	expectMatchDirect(t, Set{}, labelset)
	expectMatchDirect(t, Set{"foo": "bar"}, labelset)
	expectMatchDirect(t, Set{"baz": "blah"}, labelset)
	expectMatchDirect(t, Set{"foo": "bar", "baz": "blah"}, labelset)

	//TODO: bad values not handled for the moment in SelectorFromSet
	//expectNoMatchDirect(t, Set{"foo": "=blah"}, labelset)
	//expectNoMatchDirect(t, Set{"baz": "=bar"}, labelset)
	//expectNoMatchDirect(t, Set{"foo": "=bar", "foobar": "bar", "baz": "blah"}, labelset)
}

func TestNilMapIsValid(t *testing.T) {
	selector := Set(nil).AsSelector()
	if selector == nil {
		t.Errorf("Selector for nil set should be Everything")
	}
	if !selector.Empty() {
		t.Errorf("Selector for nil set should be Empty")
	}
}

func TestSetIsEmpty(t *testing.T) {
	if !(Set{}).AsSelector().Empty() {
		t.Errorf("Empty set should be empty")
	}
	if !(NewSelector()).Empty() {
		t.Errorf("Nil Selector should be empty")
	}
}

func TestLexer(t *testing.T) {
	testcases := []struct {
		s string
		t Token
	}{
		{"", EndOfStringToken},
		{",", CommaToken},
		{"notin", NotInToken},
		{"in", InToken},
		{"=", EqualsToken},
		{"==", DoubleEqualsToken},
		{">", GreaterThanToken},
		{"<", LessThanToken},
		//Note that Lex returns the longest valid token found
		{"!", DoesNotExistToken},
		{"!=", NotEqualsToken},
		{"(", OpenParToken},
		{")", ClosedParToken},
		//Non-"special" characters are considered part of an identifier
		{"~", IdentifierToken},
		{"||", IdentifierToken},
	}
	for _, v := range testcases {
		l := &Lexer{s: v.s, pos: 0}
		token, lit := l.Lex()
		if token != v.t {
			t.Errorf("Got %d it should be %d for '%s'", token, v.t, v.s)
		}
		if v.t != ErrorToken && lit != v.s {
			t.Errorf("Got '%s' it should be '%s'", lit, v.s)
		}
	}
}

func TestLexerSequence(t *testing.T) {
	testcases := []struct {
		s string
		t []Token
	}{
		{"key in ( value )", []Token{IdentifierToken, InToken, OpenParToken, IdentifierToken, ClosedParToken}},
		{"key notin ( value )", []Token{IdentifierToken, NotInToken, OpenParToken, IdentifierToken, ClosedParToken}},
		{"key in ( value1, value2 )", []Token{IdentifierToken, InToken, OpenParToken, IdentifierToken, CommaToken, IdentifierToken, ClosedParToken}},
		{"key", []Token{IdentifierToken}},
		{"!key", []Token{DoesNotExistToken, IdentifierToken}},
		{"()", []Token{OpenParToken, ClosedParToken}},
		{"x in (),y", []Token{IdentifierToken, InToken, OpenParToken, ClosedParToken, CommaToken, IdentifierToken}},
		{"== != (), = notin", []Token{DoubleEqualsToken, NotEqualsToken, OpenParToken, ClosedParToken, CommaToken, EqualsToken, NotInToken}},
		{"key>2", []Token{IdentifierToken, GreaterThanToken, IdentifierToken}},
		{"key<1", []Token{IdentifierToken, LessThanToken, IdentifierToken}},
	}
	for _, v := range testcases {
		var tokens []Token
		l := &Lexer{s: v.s, pos: 0}
		for {
			token, _ := l.Lex()
			if token == EndOfStringToken {
				break
			}
			tokens = append(tokens, token)
		}
		if len(tokens) != len(v.t) {
			t.Errorf("Bad number of tokens for '%s %d, %d", v.s, len(tokens), len(v.t))
		}
		for i := 0; i < min(len(tokens), len(v.t)); i++ {
			if tokens[i] != v.t[i] {
				t.Errorf("Test '%s': Mismatching in token type found '%v' it should be '%v'", v.s, tokens[i], v.t[i])
			}
		}
	}
}
func TestParserLookahead(t *testing.T) {
	testcases := []struct {
		s string
		t []Token
	}{
		{"key in ( value )", []Token{IdentifierToken, InToken, OpenParToken, IdentifierToken, ClosedParToken, EndOfStringToken}},
		{"key notin ( value )", []Token{IdentifierToken, NotInToken, OpenParToken, IdentifierToken, ClosedParToken, EndOfStringToken}},
		{"key in ( value1, value2 )", []Token{IdentifierToken, InToken, OpenParToken, IdentifierToken, CommaToken, IdentifierToken, ClosedParToken, EndOfStringToken}},
		{"key", []Token{IdentifierToken, EndOfStringToken}},
		{"!key", []Token{DoesNotExistToken, IdentifierToken, EndOfStringToken}},
		{"()", []Token{OpenParToken, ClosedParToken, EndOfStringToken}},
		{"", []Token{EndOfStringToken}},
		{"x in (),y", []Token{IdentifierToken, InToken, OpenParToken, ClosedParToken, CommaToken, IdentifierToken, EndOfStringToken}},
		{"== != (), = notin", []Token{DoubleEqualsToken, NotEqualsToken, OpenParToken, ClosedParToken, CommaToken, EqualsToken, NotInToken, EndOfStringToken}},
		{"key>2", []Token{IdentifierToken, GreaterThanToken, IdentifierToken, EndOfStringToken}},
		{"key<1", []Token{IdentifierToken, LessThanToken, IdentifierToken, EndOfStringToken}},
	}
	for _, v := range testcases {
		p := &Parser{l: &Lexer{s: v.s, pos: 0}, position: 0}
		p.scan()
		if len(p.scannedItems) != len(v.t) {
			t.Errorf("Expected %d items found %d", len(v.t), len(p.scannedItems))
		}
		for {
			token, lit := p.lookahead(KeyAndOperator)

			token2, lit2 := p.consume(KeyAndOperator)
			if token == EndOfStringToken {
				break
			}
			if token != token2 || lit != lit2 {
				t.Errorf("Bad values")
			}
		}
	}
}

func TestParseOperator(t *testing.T) {
	testcases := []struct {
		token         string
		expectedError error
	}{
		{"in", nil},
		{"=", nil},
		{"==", nil},
		{">", nil},
		{"<", nil},
		{"notin", nil},
		{"!=", nil},
		{"!", fmt.Errorf("found '%s', expected: %v", selection.DoesNotExist, strings.Join(binaryOperators, ", "))},
		{"exists", fmt.Errorf("found '%s', expected: %v", selection.Exists, strings.Join(binaryOperators, ", "))},
		{"(", fmt.Errorf("found '%s', expected: %v", "(", strings.Join(binaryOperators, ", "))},
	}
	for _, testcase := range testcases {
		p := &Parser{l: &Lexer{s: testcase.token, pos: 0}, position: 0}
		p.scan()

		_, err := p.parseOperator()
		if ok := reflect.DeepEqual(testcase.expectedError, err); !ok {
			t.Errorf("\nexpect err [%v], \nactual err [%v]", testcase.expectedError, err)
		}
	}
}

func TestRequirementConstructor(t *testing.T) {
	requirementConstructorTests := []struct {
		Key     string
		Op      selection.Operator
		Vals    sets.String
		WantErr field.ErrorList
	}{
		{
			Key: "x1",
			Op:  selection.In,
			WantErr: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeInvalid,
					Field:    "values",
					BadValue: []string{},
				},
			},
		},
		{
			Key:  "x2",
			Op:   selection.NotIn,
			Vals: sets.NewString(),
			WantErr: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeInvalid,
					Field:    "values",
					BadValue: []string{},
				},
			},
		},
		{
			Key:  "x3",
			Op:   selection.In,
			Vals: sets.NewString("foo"),
		},
		{
			Key:  "x4",
			Op:   selection.NotIn,
			Vals: sets.NewString("foo"),
		},
		{
			Key:  "x5",
			Op:   selection.Equals,
			Vals: sets.NewString("foo", "bar"),
			WantErr: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeInvalid,
					Field:    "values",
					BadValue: []string{"bar", "foo"},
				},
			},
		},
		{
			Key: "x6",
			Op:  selection.Exists,
		},
		{
			Key: "x7",
			Op:  selection.DoesNotExist,
		},
		{
			Key:  "x8",
			Op:   selection.Exists,
			Vals: sets.NewString("foo"),
			WantErr: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeInvalid,
					Field:    "values",
					BadValue: []string{"foo"},
				},
			},
		},
		{
			Key:  "x9",
			Op:   selection.In,
			Vals: sets.NewString("bar"),
		},
		{
			Key:  "x10",
			Op:   selection.In,
			Vals: sets.NewString("bar"),
		},
		{
			Key:  "x11",
			Op:   selection.GreaterThan,
			Vals: sets.NewString("1"),
		},
		{
			Key:  "x12",
			Op:   selection.LessThan,
			Vals: sets.NewString("6"),
		},
		{
			Key: "x13",
			Op:  selection.GreaterThan,
			WantErr: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeInvalid,
					Field:    "values",
					BadValue: []string{},
				},
			},
		},
		{
			Key:  "x14",
			Op:   selection.GreaterThan,
			Vals: sets.NewString("bar"),
			WantErr: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeInvalid,
					Field:    "values[0]",
					BadValue: "bar",
				},
			},
		},
		{
			Key:  "x15",
			Op:   selection.LessThan,
			Vals: sets.NewString("bar"),
			WantErr: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeInvalid,
					Field:    "values[0]",
					BadValue: "bar",
				},
			},
		},
		{
			Key: strings.Repeat("a", 254), //breaks DNS rule that len(key) <= 253
			Op:  selection.Exists,
			WantErr: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeInvalid,
					Field:    "key",
					BadValue: strings.Repeat("a", 254),
				},
			},
		},
		{
			Key:  "x16",
			Op:   selection.Equals,
			Vals: sets.NewString(strings.Repeat("a", 254)),
			WantErr: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeInvalid,
					Field:    "values[0][x16]",
					BadValue: strings.Repeat("a", 254),
				},
			},
		},
		{
			Key:  "x17",
			Op:   selection.Equals,
			Vals: sets.NewString("a b"),
			WantErr: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeInvalid,
					Field:    "values[0][x17]",
					BadValue: "a b",
				},
			},
		},
		{
			Key: "x18",
			Op:  "unsupportedOp",
			WantErr: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeNotSupported,
					Field:    "operator",
					BadValue: selection.Operator("unsupportedOp"),
				},
			},
		},
	}
	for _, rc := range requirementConstructorTests {
		_, err := NewRequirement(rc.Key, rc.Op, rc.Vals.List())
		if diff := cmp.Diff(rc.WantErr.ToAggregate(), err, ignoreDetail); diff != "" {
			t.Errorf("NewRequirement test %v returned unexpected error (-want,+got):\n%s", rc.Key, diff)
		}
	}
}

func TestToString(t *testing.T) {
	var req Requirement
	toStringTests := []struct {
		In    *internalSelector
		Out   string
		Valid bool
	}{

		{&internalSelector{
			getRequirement("x", selection.In, sets.NewString("abc", "def"), t),
			getRequirement("y", selection.NotIn, sets.NewString("jkl"), t),
			getRequirement("z", selection.Exists, nil, t)},
			"x in (abc,def),y notin (jkl),z", true},
		{&internalSelector{
			getRequirement("x", selection.NotIn, sets.NewString("abc", "def"), t),
			getRequirement("y", selection.NotEquals, sets.NewString("jkl"), t),
			getRequirement("z", selection.DoesNotExist, nil, t)},
			"x notin (abc,def),y!=jkl,!z", true},
		{&internalSelector{
			getRequirement("x", selection.In, sets.NewString("abc", "def"), t),
			req}, // adding empty req for the trailing ','
			"x in (abc,def),", false},
		{&internalSelector{
			getRequirement("x", selection.NotIn, sets.NewString("abc"), t),
			getRequirement("y", selection.In, sets.NewString("jkl", "mno"), t),
			getRequirement("z", selection.NotIn, sets.NewString(""), t)},
			"x notin (abc),y in (jkl,mno),z notin ()", true},
		{&internalSelector{
			getRequirement("x", selection.Equals, sets.NewString("abc"), t),
			getRequirement("y", selection.DoubleEquals, sets.NewString("jkl"), t),
			getRequirement("z", selection.NotEquals, sets.NewString("a"), t),
			getRequirement("z", selection.Exists, nil, t)},
			"x=abc,y==jkl,z!=a,z", true},
		{&internalSelector{
			getRequirement("x", selection.GreaterThan, sets.NewString("2"), t),
			getRequirement("y", selection.LessThan, sets.NewString("8"), t),
			getRequirement("z", selection.Exists, nil, t)},
			"x>2,y<8,z", true},
	}
	for _, ts := range toStringTests {
		if out := ts.In.String(); out == "" && ts.Valid {
			t.Errorf("%#v.String() => '%v' expected no error", ts.In, out)
		} else if out != ts.Out {
			t.Errorf("%#v.String() => '%v' want '%v'", ts.In, out, ts.Out)
		}
	}
}

func TestRequirementSelectorMatching(t *testing.T) {
	var req Requirement
	labelSelectorMatchingTests := []struct {
		Set   Set
		Sel   Selector
		Match bool
	}{
		{Set{"x": "foo", "y": "baz"}, &internalSelector{
			req,
		}, false},
		{Set{"x": "foo", "y": "baz"}, &internalSelector{
			getRequirement("x", selection.In, sets.NewString("foo"), t),
			getRequirement("y", selection.NotIn, sets.NewString("alpha"), t),
		}, true},
		{Set{"x": "foo", "y": "baz"}, &internalSelector{
			getRequirement("x", selection.In, sets.NewString("foo"), t),
			getRequirement("y", selection.In, sets.NewString("alpha"), t),
		}, false},
		{Set{"y": ""}, &internalSelector{
			getRequirement("x", selection.NotIn, sets.NewString(""), t),
			getRequirement("y", selection.Exists, nil, t),
		}, true},
		{Set{"y": ""}, &internalSelector{
			getRequirement("x", selection.DoesNotExist, nil, t),
			getRequirement("y", selection.Exists, nil, t),
		}, true},
		{Set{"y": ""}, &internalSelector{
			getRequirement("x", selection.NotIn, sets.NewString(""), t),
			getRequirement("y", selection.DoesNotExist, nil, t),
		}, false},
		{Set{"y": "baz"}, &internalSelector{
			getRequirement("x", selection.In, sets.NewString(""), t),
		}, false},
		{Set{"z": "2"}, &internalSelector{
			getRequirement("z", selection.GreaterThan, sets.NewString("1"), t),
		}, true},
		{Set{"z": "v2"}, &internalSelector{
			getRequirement("z", selection.GreaterThan, sets.NewString("1"), t),
		}, false},
	}
	for _, lsm := range labelSelectorMatchingTests {
		if match := lsm.Sel.Matches(lsm.Set); match != lsm.Match {
			t.Errorf("%+v.Matches(%#v) => %v, want %v", lsm.Sel, lsm.Set, match, lsm.Match)
		}
	}
}

func TestSetSelectorParser(t *testing.T) {
	setSelectorParserTests := []struct {
		In    string
		Out   Selector
		Match bool
		Valid bool
	}{
		{"", NewSelector(), true, true},
		{"\rx", internalSelector{
			getRequirement("x", selection.Exists, nil, t),
		}, true, true},
		{"this-is-a-dns.domain.com/key-with-dash", internalSelector{
			getRequirement("this-is-a-dns.domain.com/key-with-dash", selection.Exists, nil, t),
		}, true, true},
		{"this-is-another-dns.domain.com/key-with-dash in (so,what)", internalSelector{
			getRequirement("this-is-another-dns.domain.com/key-with-dash", selection.In, sets.NewString("so", "what"), t),
		}, true, true},
		{"0.1.2.domain/99 notin (10.10.100.1, tick.tack.clock)", internalSelector{
			getRequirement("0.1.2.domain/99", selection.NotIn, sets.NewString("10.10.100.1", "tick.tack.clock"), t),
		}, true, true},
		{"foo  in	 (abc)", internalSelector{
			getRequirement("foo", selection.In, sets.NewString("abc"), t),
		}, true, true},
		{"x notin\n (abc)", internalSelector{
			getRequirement("x", selection.NotIn, sets.NewString("abc"), t),
		}, true, true},
		{"x  notin	\t	(abc,def)", internalSelector{
			getRequirement("x", selection.NotIn, sets.NewString("abc", "def"), t),
		}, true, true},
		{"x in (abc,def)", internalSelector{
			getRequirement("x", selection.In, sets.NewString("abc", "def"), t),
		}, true, true},
		{"x in (abc,)", internalSelector{
			getRequirement("x", selection.In, sets.NewString("abc", ""), t),
		}, true, true},
		{"x in ()", internalSelector{
			getRequirement("x", selection.In, sets.NewString(""), t),
		}, true, true},
		{"x notin (abc,,def),bar,z in (),w", internalSelector{
			getRequirement("bar", selection.Exists, nil, t),
			getRequirement("w", selection.Exists, nil, t),
			getRequirement("x", selection.NotIn, sets.NewString("abc", "", "def"), t),
			getRequirement("z", selection.In, sets.NewString(""), t),
		}, true, true},
		{"x,y in (a)", internalSelector{
			getRequirement("y", selection.In, sets.NewString("a"), t),
			getRequirement("x", selection.Exists, nil, t),
		}, false, true},
		{"x=a", internalSelector{
			getRequirement("x", selection.Equals, sets.NewString("a"), t),
		}, true, true},
		{"x>1", internalSelector{
			getRequirement("x", selection.GreaterThan, sets.NewString("1"), t),
		}, true, true},
		{"x<7", internalSelector{
			getRequirement("x", selection.LessThan, sets.NewString("7"), t),
		}, true, true},
		{"x=a,y!=b", internalSelector{
			getRequirement("x", selection.Equals, sets.NewString("a"), t),
			getRequirement("y", selection.NotEquals, sets.NewString("b"), t),
		}, true, true},
		{"x=a,y!=b,z in (h,i,j)", internalSelector{
			getRequirement("x", selection.Equals, sets.NewString("a"), t),
			getRequirement("y", selection.NotEquals, sets.NewString("b"), t),
			getRequirement("z", selection.In, sets.NewString("h", "i", "j"), t),
		}, true, true},
		{"x=a||y=b", internalSelector{}, false, false},
		{"x,,y", nil, true, false},
		{",x,y", nil, true, false},
		{"x nott in (y)", nil, true, false},
		{"x notin ( )", internalSelector{
			getRequirement("x", selection.NotIn, sets.NewString(""), t),
		}, true, true},
		{"x notin (, a)", internalSelector{
			getRequirement("x", selection.NotIn, sets.NewString("", "a"), t),
		}, true, true},
		{"a in (xyz),", nil, true, false},
		{"a in (xyz)b notin ()", nil, true, false},
		{"a ", internalSelector{
			getRequirement("a", selection.Exists, nil, t),
		}, true, true},
		{"a in (x,y,notin, z,in)", internalSelector{
			getRequirement("a", selection.In, sets.NewString("in", "notin", "x", "y", "z"), t),
		}, true, true}, // operator 'in' inside list of identifiers
		{"a in (xyz abc)", nil, false, false}, // no comma
		{"a notin(", nil, true, false},        // bad formed
		{"a (", nil, false, false},            // cpar
		{"(", nil, false, false},              // opar
	}

	for _, ssp := range setSelectorParserTests {
		if sel, err := Parse(ssp.In); err != nil && ssp.Valid {
			t.Errorf("Parse(%s) => %v expected no error", ssp.In, err)
		} else if err == nil && !ssp.Valid {
			t.Errorf("Parse(%s) => %+v expected error", ssp.In, sel)
		} else if ssp.Match && !reflect.DeepEqual(sel, ssp.Out) {
			t.Errorf("Parse(%s) => parse output '%#v' doesn't match '%#v' expected match", ssp.In, sel, ssp.Out)
		}
	}
}

func getRequirement(key string, op selection.Operator, vals sets.String, t *testing.T) Requirement {
	req, err := NewRequirement(key, op, vals.List())
	if err != nil {
		t.Errorf("NewRequirement(%v, %v, %v) resulted in error:%v", key, op, vals, err)
		return Requirement{}
	}
	return *req
}

func TestAdd(t *testing.T) {
	testCases := []struct {
		name        string
		sel         Selector
		key         string
		operator    selection.Operator
		values      []string
		refSelector Selector
	}{
		{
			"keyInOperator",
			internalSelector{},
			"key",
			selection.In,
			[]string{"value"},
			internalSelector{Requirement{"key", selection.In, []string{"value"}}},
		},
		{
			"keyEqualsOperator",
			internalSelector{Requirement{"key", selection.In, []string{"value"}}},
			"key2",
			selection.Equals,
			[]string{"value2"},
			internalSelector{
				Requirement{"key", selection.In, []string{"value"}},
				Requirement{"key2", selection.Equals, []string{"value2"}},
			},
		},
	}
	for _, ts := range testCases {
		req, err := NewRequirement(ts.key, ts.operator, ts.values)
		if err != nil {
			t.Errorf("%s - Unable to create labels.Requirement", ts.name)
		}
		ts.sel = ts.sel.Add(*req)
		if !reflect.DeepEqual(ts.sel, ts.refSelector) {
			t.Errorf("%s - Expected %v found %v", ts.name, ts.refSelector, ts.sel)
		}
	}
}

func TestSafeSort(t *testing.T) {
	tests := []struct {
		name   string
		in     []string
		inCopy []string
		want   []string
	}{
		{
			name:   "nil strings",
			in:     nil,
			inCopy: nil,
			want:   nil,
		},
		{
			name:   "ordered strings",
			in:     []string{"bar", "foo"},
			inCopy: []string{"bar", "foo"},
			want:   []string{"bar", "foo"},
		},
		{
			name:   "unordered strings",
			in:     []string{"foo", "bar"},
			inCopy: []string{"foo", "bar"},
			want:   []string{"bar", "foo"},
		},
		{
			name:   "duplicated strings",
			in:     []string{"foo", "bar", "foo", "bar"},
			inCopy: []string{"foo", "bar", "foo", "bar"},
			want:   []string{"bar", "bar", "foo", "foo"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := safeSort(tt.in); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("safeSort() = %v, want %v", got, tt.want)
			}
			if !reflect.DeepEqual(tt.in, tt.inCopy) {
				t.Errorf("after safeSort(), input = %v, want %v", tt.in, tt.inCopy)
			}
		})
	}
}

func BenchmarkSelectorFromValidatedSet(b *testing.B) {
	set := map[string]string{
		"foo": "foo",
		"bar": "bar",
	}
	matchee := Set(map[string]string{
		"foo":   "foo",
		"bar":   "bar",
		"extra": "label",
	})

	for i := 0; i < b.N; i++ {
		s := SelectorFromValidatedSet(set)
		if s.Empty() {
			b.Errorf("Unexpected selector")
		}
		if !s.Matches(matchee) {
			b.Errorf("Unexpected match")
		}
	}
}

func BenchmarkSetSelector(b *testing.B) {
	set := map[string]string{
		"foo": "foo",
		"bar": "bar",
	}
	matchee := Set(map[string]string{
		"foo":   "foo",
		"bar":   "bar",
		"extra": "label",
	})

	for i := 0; i < b.N; i++ {
		s := ValidatedSetSelector(set)
		if s.Empty() {
			b.Errorf("Unexpected selector")
		}
		if !s.Matches(matchee) {
			b.Errorf("Unexpected match")
		}
	}
}

func TestSetSelectorString(t *testing.T) {
	cases := []struct {
		set Set
		out string
	}{
		{
			Set{},
			"",
		},
		{
			Set{"app": "foo"},
			"app=foo",
		},
		{
			Set{"app": "foo", "a": "b"},
			"a=b,app=foo",
		},
	}

	for _, tt := range cases {
		t.Run(tt.out, func(t *testing.T) {
			if got := ValidatedSetSelector(tt.set).String(); tt.out != got {
				t.Fatalf("expected %v, got %v", tt.out, got)
			}
		})
	}
}

func TestRequiresExactMatch(t *testing.T) {
	testCases := []struct {
		name          string
		sel           Selector
		label         string
		expectedFound bool
		expectedValue string
	}{
		{
			name:          "keyInOperatorExactMatch",
			sel:           internalSelector{Requirement{"key", selection.In, []string{"value"}}},
			label:         "key",
			expectedFound: true,
			expectedValue: "value",
		},
		{
			name:          "keyInOperatorNotExactMatch",
			sel:           internalSelector{Requirement{"key", selection.In, []string{"value", "value2"}}},
			label:         "key",
			expectedFound: false,
			expectedValue: "",
		},
		{
			name: "keyInOperatorNotExactMatch",
			sel: internalSelector{
				Requirement{"key", selection.In, []string{"value", "value1"}},
				Requirement{"key2", selection.In, []string{"value2"}},
			},
			label:         "key2",
			expectedFound: true,
			expectedValue: "value2",
		},
		{
			name:          "keyEqualOperatorExactMatch",
			sel:           internalSelector{Requirement{"key", selection.Equals, []string{"value"}}},
			label:         "key",
			expectedFound: true,
			expectedValue: "value",
		},
		{
			name:          "keyDoubleEqualOperatorExactMatch",
			sel:           internalSelector{Requirement{"key", selection.DoubleEquals, []string{"value"}}},
			label:         "key",
			expectedFound: true,
			expectedValue: "value",
		},
		{
			name:          "keyNotEqualOperatorExactMatch",
			sel:           internalSelector{Requirement{"key", selection.NotEquals, []string{"value"}}},
			label:         "key",
			expectedFound: false,
			expectedValue: "",
		},
		{
			name: "keyEqualOperatorExactMatchFirst",
			sel: internalSelector{
				Requirement{"key", selection.In, []string{"value"}},
				Requirement{"key2", selection.In, []string{"value2"}},
			},
			label:         "key",
			expectedFound: true,
			expectedValue: "value",
		},
	}
	for _, ts := range testCases {
		t.Run(ts.name, func(t *testing.T) {
			value, found := ts.sel.RequiresExactMatch(ts.label)
			if found != ts.expectedFound {
				t.Errorf("Expected match %v, found %v", ts.expectedFound, found)
			}
			if found && value != ts.expectedValue {
				t.Errorf("Expected value %v, found %v", ts.expectedValue, value)
			}

		})
	}
}

func TestValidatedSelectorFromSet(t *testing.T) {
	tests := []struct {
		name             string
		input            Set
		expectedSelector internalSelector
		expectedError    field.ErrorList
	}{
		{
			name:  "Simple Set, no error",
			input: Set{"key": "val"},
			expectedSelector: internalSelector{
				Requirement{
					key:       "key",
					operator:  selection.Equals,
					strValues: []string{"val"},
				},
			},
		},
		{
			name:  "Invalid Set, value too long",
			input: Set{"Key": "axahm2EJ8Phiephe2eixohbee9eGeiyees1thuozi1xoh0GiuH3diewi8iem7Nui"},
			expectedError: field.ErrorList{
				&field.Error{
					Type:     field.ErrorTypeInvalid,
					Field:    "values[0][Key]",
					BadValue: "axahm2EJ8Phiephe2eixohbee9eGeiyees1thuozi1xoh0GiuH3diewi8iem7Nui",
				},
			},
		},
	}

	for _, tc := range tests {
		selector, err := ValidatedSelectorFromSet(tc.input)
		if diff := cmp.Diff(tc.expectedError.ToAggregate(), err, ignoreDetail); diff != "" {
			t.Errorf("ValidatedSelectorFromSet %#v returned unexpected error (-want,+got):\n%s", tc.name, diff)
		}
		if err == nil {
			if diff := cmp.Diff(tc.expectedSelector, selector); diff != "" {
				t.Errorf("ValidatedSelectorFromSet %#v returned unexpected selector (-want,+got):\n%s", tc.name, diff)
			}
		}
	}
}

func BenchmarkRequirementString(b *testing.B) {
	r := Requirement{
		key:      "environment",
		operator: selection.NotIn,
		strValues: []string{
			"dev",
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if r.String() != "environment notin (dev)" {
			b.Errorf("Unexpected Requirement string")
		}
	}
}

func BenchmarkRequirementMatches(b *testing.B) {
	r := Requirement{
		key:      "environment",
		operator: selection.NotIn,
		strValues: []string{
			"dev",
		},
	}
	labels := Set(map[string]string{
		"key":         "value",
		"environment": "dev",
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Matches(labels)
	}
}

func TestRequirementEqual(t *testing.T) {
	tests := []struct {
		name string
		x, y *Requirement
		want bool
	}{
		{
			name: "same requirements should be equal",
			x: &Requirement{
				key:       "key",
				operator:  selection.Equals,
				strValues: []string{"foo", "bar"},
			},
			y: &Requirement{
				key:       "key",
				operator:  selection.Equals,
				strValues: []string{"foo", "bar"},
			},
			want: true,
		},
		{
			name: "requirements with different keys should not be equal",
			x: &Requirement{
				key:       "key1",
				operator:  selection.Equals,
				strValues: []string{"foo", "bar"},
			},
			y: &Requirement{
				key:       "key2",
				operator:  selection.Equals,
				strValues: []string{"foo", "bar"},
			},
			want: false,
		},
		{
			name: "requirements with different operators should not be equal",
			x: &Requirement{
				key:       "key",
				operator:  selection.Equals,
				strValues: []string{"foo", "bar"},
			},
			y: &Requirement{
				key:       "key",
				operator:  selection.In,
				strValues: []string{"foo", "bar"},
			},
			want: false,
		},
		{
			name: "requirements with different values should not be equal",
			x: &Requirement{
				key:       "key",
				operator:  selection.Equals,
				strValues: []string{"foo", "bar"},
			},
			y: &Requirement{
				key:       "key",
				operator:  selection.Equals,
				strValues: []string{"foobar"},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cmp.Equal(tt.x, tt.y); got != tt.want {
				t.Errorf("cmp.Equal() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatchesNothing(t *testing.T) {
	tests := []struct {
		name          string
		selector      string
		set           map[string]string
		labelSelector Selector
		want          bool
	}{
		{
			name:          "MatchNothing should match Nothing()",
			labelSelector: Nothing(),
			want:          true,
		},
		{
			name:          "MatchNothing should match sharedNothingSelector",
			labelSelector: sharedNothingSelector,
			want:          true,
		},
		{
			name:          "MatchNothing should not match Everything()",
			labelSelector: Everything(),
			want:          false,
		},
		{
			name:          "MatchNothing should not match sharedEverythingSelector",
			labelSelector: sharedEverythingSelector,
			want:          false,
		},
		{
			name: "MatchNothing should not match empty set",
			set:  map[string]string{},
			want: false,
		},
		{
			name: "MatchNothing should not match non-empty set",
			set:  map[string]string{"key": "value"},
			want: false,
		},
		{
			name:     "MatchNothing should not match empty selector",
			selector: "",
			want:     false,
		},
		{
			name:     "MatchNothing should not match non-empty selector - exists",
			selector: "a",
			want:     false,
		},
		{
			name:     "MatchNothing should not match non-empty selector - not exists",
			selector: "!a",
			want:     false,
		},
		{
			name:     "MatchNothing should not match non-empty selector - equals",
			selector: "a=b",
			want:     false,
		},
		{
			name:     "MatchNothing should not match non-empty selector - not equals",
			selector: "a!=b",
			want:     false,
		},
		{
			name:     "MatchNothing should not match non-empty selector - in",
			selector: "a in (b)",
			want:     false,
		},
		{
			name:     "MatchNothing should not match non-empty selector - notin",
			selector: "a notin (b)",
			want:     false,
		},
		{
			name:     "MatchNothing should not match non-empty selector - conflict exists and not exists",
			selector: "a,!a",
			want:     false,
		},
		{
			name:     "MatchNothing should not match non-empty selector - conflict equals and not equals",
			selector: "a=b,a!=b",
			want:     false,
		},
		{
			name:     "MatchNothing should not match non-empty selector - conflict in and notin",
			selector: "a in (b),a notin (b)",
			want:     false,
		},
	}

	for i := 0; i < len(tests); i++ {
		if tests[i].labelSelector != nil {
			expectMatchNothing(t, tests[i].labelSelector, tests[i].want)
		} else if tests[i].set != nil {
			expectMatchNothing(t, SelectorFromSet(tests[i].set), tests[i].want)
		} else {
			selector, err := Parse(tests[i].selector)
			if err != nil {
				t.Errorf("Unable to parse %v as a selector.\n", selector)
			}
			expectMatchNothing(t, selector, tests[i].want)
		}
	}
}

func expectMatchNothing(t *testing.T, selector Selector, want bool) {
	if MatchesNothing(selector) != want {
		t.Errorf("Wanted %s to MatchNothing '%t', but it did not.\n", selector, want)
	}
}
