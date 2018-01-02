package merkletrie_test

import (
	"bytes"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"
	"unicode"

	"gopkg.in/src-d/go-git.v4/utils/merkletrie"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie/internal/fsnoder"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type DiffTreeSuite struct{}

var _ = Suite(&DiffTreeSuite{})

type diffTreeTest struct {
	from     string
	to       string
	expected string
}

func (t diffTreeTest) innerRun(c *C, context string, reverse bool) {
	comment := Commentf("\n%s", context)
	if reverse {
		comment = Commentf("%s [REVERSED]", comment.CheckCommentString())
	}

	a, err := fsnoder.New(t.from)
	c.Assert(err, IsNil, comment)
	comment = Commentf("%s\n\t    from = %s", comment.CheckCommentString(), a)

	b, err := fsnoder.New(t.to)
	c.Assert(err, IsNil, comment)
	comment = Commentf("%s\n\t      to = %s", comment.CheckCommentString(), b)

	expected, err := newChangesFromString(t.expected)
	c.Assert(err, IsNil, comment)

	if reverse {
		a, b = b, a
		expected = expected.reverse()
	}
	comment = Commentf("%s\n\texpected = %s", comment.CheckCommentString(), expected)

	results, err := merkletrie.DiffTree(a, b, fsnoder.HashEqual)
	c.Assert(err, IsNil, comment)

	obtained, err := newChanges(results)
	c.Assert(err, IsNil, comment)

	comment = Commentf("%s\n\tobtained = %s", comment.CheckCommentString(), obtained)

	c.Assert(obtained, changesEquals, expected, comment)
}

func (t diffTreeTest) run(c *C, context string) {
	t.innerRun(c, context, false)
	t.innerRun(c, context, true)
}

type change struct {
	merkletrie.Action
	path string
}

func (c change) String() string {
	return fmt.Sprintf("<%s %s>", c.Action, c.path)
}

func (c change) reverse() change {
	ret := change{
		path: c.path,
	}

	switch c.Action {
	case merkletrie.Insert:
		ret.Action = merkletrie.Delete
	case merkletrie.Delete:
		ret.Action = merkletrie.Insert
	case merkletrie.Modify:
		ret.Action = merkletrie.Modify
	default:
		panic(fmt.Sprintf("unknown action type: %d", c.Action))
	}

	return ret
}

type changes []change

func newChanges(original merkletrie.Changes) (changes, error) {
	ret := make(changes, len(original))
	for i, c := range original {
		action, err := c.Action()
		if err != nil {
			return nil, err
		}
		switch action {
		case merkletrie.Insert:
			ret[i] = change{
				Action: merkletrie.Insert,
				path:   c.To.String(),
			}
		case merkletrie.Delete:
			ret[i] = change{
				Action: merkletrie.Delete,
				path:   c.From.String(),
			}
		case merkletrie.Modify:
			ret[i] = change{
				Action: merkletrie.Modify,
				path:   c.From.String(),
			}
		default:
			panic(fmt.Sprintf("unsupported action %d", action))
		}
	}

	return ret, nil
}

func newChangesFromString(s string) (changes, error) {
	ret := make([]change, 0)

	s = strings.TrimSpace(s)
	s = removeDuplicatedSpace(s)
	s = turnSpaceIntoLiteralSpace(s)

	if s == "" {
		return ret, nil
	}

	for _, chunk := range strings.Split(s, " ") {
		change := change{
			path: string(chunk[1:]),
		}

		switch chunk[0] {
		case '+':
			change.Action = merkletrie.Insert
		case '-':
			change.Action = merkletrie.Delete
		case '*':
			change.Action = merkletrie.Modify
		default:
			panic(fmt.Sprintf("unsupported action descriptor %q", chunk[0]))
		}

		ret = append(ret, change)
	}

	return ret, nil
}

func removeDuplicatedSpace(s string) string {
	var buf bytes.Buffer

	var lastWasSpace, currentIsSpace bool
	for _, r := range s {
		currentIsSpace = unicode.IsSpace(r)

		if lastWasSpace && currentIsSpace {
			continue
		}
		lastWasSpace = currentIsSpace

		buf.WriteRune(r)
	}

	return buf.String()
}

func turnSpaceIntoLiteralSpace(s string) string {
	return strings.Map(
		func(r rune) rune {
			if unicode.IsSpace(r) {
				return ' '
			}
			return r
		}, s)
}

func (cc changes) Len() int           { return len(cc) }
func (cc changes) Swap(i, j int)      { cc[i], cc[j] = cc[j], cc[i] }
func (cc changes) Less(i, j int) bool { return strings.Compare(cc[i].String(), cc[j].String()) < 0 }

func (cc changes) equals(other changes) bool {
	sort.Sort(cc)
	sort.Sort(other)
	return reflect.DeepEqual(cc, other)
}

func (cc changes) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "len(%d) [", len(cc))
	sep := ""
	for _, c := range cc {
		fmt.Fprintf(&buf, "%s%s", sep, c)
		sep = ", "
	}
	buf.WriteByte(']')
	return buf.String()
}

func (cc changes) reverse() changes {
	ret := make(changes, len(cc))
	for i, c := range cc {
		ret[i] = c.reverse()
	}

	return ret
}

type changesEqualsChecker struct {
	*CheckerInfo
}

var changesEquals Checker = &changesEqualsChecker{
	&CheckerInfo{Name: "changesEquals", Params: []string{"obtained", "expected"}},
}

func (checker *changesEqualsChecker) Check(params []interface{}, names []string) (result bool, error string) {
	a, ok := params[0].(changes)
	if !ok {
		return false, "first parameter must be a changes"
	}
	b, ok := params[1].(changes)
	if !ok {
		return false, "second parameter must be a changes"
	}

	return a.equals(b), ""
}

func do(c *C, list []diffTreeTest) {
	for i, t := range list {
		t.run(c, fmt.Sprintf("test #%d:", i))
	}
}

func (s *DiffTreeSuite) TestEmptyVsEmpty(c *C) {
	do(c, []diffTreeTest{
		{"()", "()", ""},
		{"A()", "A()", ""},
		{"A()", "()", ""},
		{"A()", "B()", ""},
	})
}

func (s *DiffTreeSuite) TestBasicCases(c *C) {
	do(c, []diffTreeTest{
		{"()", "()", ""},
		{"()", "(a<>)", "+a"},
		{"()", "(a<1>)", "+a"},
		{"()", "(a())", ""},
		{"()", "(a(b()))", ""},
		{"()", "(a(b<>))", "+a/b"},
		{"()", "(a(b<1>))", "+a/b"},
		{"(a<>)", "(a<>)", ""},
		{"(a<>)", "(a<1>)", "*a"},
		{"(a<>)", "(a())", "-a"},
		{"(a<>)", "(a(b()))", "-a"},
		{"(a<>)", "(a(b<>))", "-a +a/b"},
		{"(a<>)", "(a(b<1>))", "-a +a/b"},
		{"(a<>)", "(c())", "-a"},
		{"(a<>)", "(c(b()))", "-a"},
		{"(a<>)", "(c(b<>))", "-a +c/b"},
		{"(a<>)", "(c(b<1>))", "-a +c/b"},
		{"(a<>)", "(c(a()))", "-a"},
		{"(a<>)", "(c(a<>))", "-a +c/a"},
		{"(a<>)", "(c(a<1>))", "-a +c/a"},
		{"(a<1>)", "(a<1>)", ""},
		{"(a<1>)", "(a<2>)", "*a"},
		{"(a<1>)", "(b<1>)", "-a +b"},
		{"(a<1>)", "(b<2>)", "-a +b"},
		{"(a<1>)", "(a())", "-a"},
		{"(a<1>)", "(a(b()))", "-a"},
		{"(a<1>)", "(a(b<>))", "-a +a/b"},
		{"(a<1>)", "(a(b<1>))", "-a +a/b"},
		{"(a<1>)", "(a(b<2>))", "-a +a/b"},
		{"(a<1>)", "(c())", "-a"},
		{"(a<1>)", "(c(b()))", "-a"},
		{"(a<1>)", "(c(b<>))", "-a +c/b"},
		{"(a<1>)", "(c(b<1>))", "-a +c/b"},
		{"(a<1>)", "(c(b<2>))", "-a +c/b"},
		{"(a<1>)", "(c())", "-a"},
		{"(a<1>)", "(c(a()))", "-a"},
		{"(a<1>)", "(c(a<>))", "-a +c/a"},
		{"(a<1>)", "(c(a<1>))", "-a +c/a"},
		{"(a<1>)", "(c(a<2>))", "-a +c/a"},
		{"(a())", "(a())", ""},
		{"(a())", "(b())", ""},
		{"(a())", "(a(b()))", ""},
		{"(a())", "(b(a()))", ""},
		{"(a())", "(a(b<>))", "+a/b"},
		{"(a())", "(a(b<1>))", "+a/b"},
		{"(a())", "(b(a<>))", "+b/a"},
		{"(a())", "(b(a<1>))", "+b/a"},
	})
}

func (s *DiffTreeSuite) TestHorizontals(c *C) {
	do(c, []diffTreeTest{
		{"()", "(a<> b<>)", "+a +b"},
		{"()", "(a<> b<1>)", "+a +b"},
		{"()", "(a<> b())", "+a"},
		{"()", "(a() b<>)", "+b"},
		{"()", "(a<1> b<>)", "+a +b"},
		{"()", "(a<1> b<1>)", "+a +b"},
		{"()", "(a<1> b<2>)", "+a +b"},
		{"()", "(a<1> b())", "+a"},
		{"()", "(a() b<1>)", "+b"},
		{"()", "(a() b())", ""},
		{"()", "(a<> b<> c<> d<>)", "+a +b +c +d"},
		{"()", "(a<> b<1> c() d<> e<2> f())", "+a +b +d +e"},
	})
}

func (s *DiffTreeSuite) TestVerticals(c *C) {
	do(c, []diffTreeTest{
		{"()", "(z<>)", "+z"},
		{"()", "(a(z<>))", "+a/z"},
		{"()", "(a(b(z<>)))", "+a/b/z"},
		{"()", "(a(b(c(z<>))))", "+a/b/c/z"},
		{"()", "(a(b(c(d(z<>)))))", "+a/b/c/d/z"},
		{"()", "(a(b(c(d(z<1>)))))", "+a/b/c/d/z"},
	})
}

func (s *DiffTreeSuite) TestSingleInserts(c *C) {
	do(c, []diffTreeTest{
		{"()", "(z<>)", "+z"},
		{"(a())", "(a(z<>))", "+a/z"},
		{"(a())", "(a(b(z<>)))", "+a/b/z"},
		{"(a(b(c())))", "(a(b(c(z<>))))", "+a/b/c/z"},
		{"(a<> b<> c<>)", "(a<> b<> c<> z<>)", "+z"},
		{"(a(b<> c<> d<>))", "(a(b<> c<> d<> z<>))", "+a/z"},
		{"(a(b(c<> d<> e<>)))", "(a(b(c<> d<> e<> z<>)))", "+a/b/z"},
		{"(a(b<>) f<>)", "(a(b<>) f<> z<>)", "+z"},
		{"(a(b<>) f<>)", "(a(b<> z<>) f<>)", "+a/z"},
	})
}

func (s *DiffTreeSuite) TestDebug(c *C) {
	do(c, []diffTreeTest{
		{"(a(b<>) f<>)", "(a(b<> z<>) f<>)", "+a/z"},
	})
}

//      root
//      / | \
//     /  |  ----
//    f   d      h --------
//   /\         /  \      |
//  e   a      j   b/      g
//  |  / \     |
//  l  n  k    icm
//     |
//     o
//     |
//     p/
func (s *DiffTreeSuite) TestCrazy(c *C) {
	crazy := "(f(e(l<1>) a(n(o(p())) k<1>)) d<1> h(j(i<1> c<2> m<>) b() g<>))"
	do(c, []diffTreeTest{
		{
			crazy,
			"()",
			"-d -f/e/l -f/a/k -h/j/i -h/j/c -h/j/m -h/g",
		}, {
			crazy,
			crazy,
			"",
		}, {
			crazy,
			"(d<1>)",
			"-f/e/l -f/a/k -h/j/i -h/j/c -h/j/m -h/g",
		}, {
			crazy,
			"(d<1> h(b() g<>))",
			"-f/e/l -f/a/k -h/j/i -h/j/c -h/j/m",
		}, {
			crazy,
			"(d<1> f(e(l()) a()) h(b() g<>))",
			"-f/e/l -f/a/k -h/j/i -h/j/c -h/j/m",
		}, {
			crazy,
			"(d<1> f(e(l<1>) a()) h(b() g<>))",
			"-f/a/k -h/j/i -h/j/c -h/j/m",
		}, {
			crazy,
			"(d<2> f(e(l<2>) a(s(t<1>))) h(b() g<> r<> j(i<> c<3> m<>)))",
			"+f/a/s/t +h/r -f/a/k *d *f/e/l *h/j/c *h/j/i",
		}, {
			crazy,
			"(f(e(l<2>) a(n(o(p<1>)) k<>)) h(j(i<1> c<2> m<>) b() g<>))",
			"*f/e/l +f/a/n/o/p *f/a/k -d",
		}, {
			crazy,
			"(f(e(l<1>) a(n(o(p(r<1>))) k<1>)) d<1> h(j(i<1> c<2> b() m<>) g<1>))",
			"+f/a/n/o/p/r *h/g",
		},
	})
}

func (s *DiffTreeSuite) TestSameNames(c *C) {
	do(c, []diffTreeTest{
		{
			"(a(a(a<>)))",
			"(a(a(a<1>)))",
			"*a/a/a",
		}, {
			"(a(b(a<>)))",
			"(a(b(a<>)) b(a<>))",
			"+b/a",
		}, {
			"(a(b(a<>)))",
			"(a(b()) b(a<>))",
			"-a/b/a +b/a",
		},
	})
}

func (s *DiffTreeSuite) TestIssue275(c *C) {
	do(c, []diffTreeTest{
		{
			"(a(b(c.go<1>) b.go<2>))",
			"(a(b(c.go<1> d.go<3>) b.go<2>))",
			"+a/b/d.go",
		},
	})
}
