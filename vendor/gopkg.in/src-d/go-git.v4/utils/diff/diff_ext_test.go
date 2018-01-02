package diff_test

import (
	"testing"

	"gopkg.in/src-d/go-git.v4/utils/diff"

	"github.com/sergi/go-diff/diffmatchpatch"
	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type suiteCommon struct{}

var _ = Suite(&suiteCommon{})

var diffTests = [...]struct {
	src string // the src string to diff
	dst string // the dst string to diff
}{
	// equal inputs
	{"", ""},
	{"a", "a"},
	{"a\n", "a\n"},
	{"a\nb", "a\nb"},
	{"a\nb\n", "a\nb\n"},
	{"a\nb\nc", "a\nb\nc"},
	{"a\nb\nc\n", "a\nb\nc\n"},
	// missing '\n'
	{"", "\n"},
	{"\n", ""},
	{"a", "a\n"},
	{"a\n", "a"},
	{"a\nb", "a\nb"},
	{"a\nb\n", "a\nb\n"},
	{"a\nb\nc", "a\nb\nc"},
	{"a\nb\nc\n", "a\nb\nc\n"},
	// generic
	{"a\nbbbbb\n\tccc\ndd\n\tfffffffff\n", "bbbbb\n\tccc\n\tDD\n\tffff\n"},
}

func (s *suiteCommon) TestAll(c *C) {
	for i, t := range diffTests {
		diffs := diff.Do(t.src, t.dst)
		src := diff.Src(diffs)
		dst := diff.Dst(diffs)
		c.Assert(src, Equals, t.src, Commentf("subtest %d, src=%q, dst=%q, bad calculated src", i, t.src, t.dst))
		c.Assert(dst, Equals, t.dst, Commentf("subtest %d, src=%q, dst=%q, bad calculated dst", i, t.src, t.dst))
	}
}

var doTests = [...]struct {
	src, dst string
	exp      []diffmatchpatch.Diff
}{
	{
		src: "",
		dst: "",
		exp: []diffmatchpatch.Diff{},
	},
	{
		src: "a",
		dst: "a",
		exp: []diffmatchpatch.Diff{
			{
				Type: 0,
				Text: "a",
			},
		},
	},
	{
		src: "",
		dst: "abc\ncba",
		exp: []diffmatchpatch.Diff{
			{
				Type: 1,
				Text: "abc\ncba",
			},
		},
	},
	{
		src: "abc\ncba",
		dst: "",
		exp: []diffmatchpatch.Diff{
			{
				Type: -1,
				Text: "abc\ncba",
			},
		},
	},
	{
		src: "abc\nbcd\ncde",
		dst: "000\nabc\n111\nBCD\n",
		exp: []diffmatchpatch.Diff{
			{Type: 1, Text: "000\n"},
			{Type: 0, Text: "abc\n"},
			{Type: -1, Text: "bcd\ncde"},
			{Type: 1, Text: "111\nBCD\n"},
		},
	},
}

func (s *suiteCommon) TestDo(c *C) {
	for i, t := range doTests {
		diffs := diff.Do(t.src, t.dst)
		c.Assert(diffs, DeepEquals, t.exp, Commentf("subtest %d", i))
	}
}
