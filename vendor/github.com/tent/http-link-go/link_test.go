package link

import (
	"testing"

	. "launchpad.net/gocheck"
)

// Hook up gocheck into the "go test" runner.
func Test(t *testing.T) { TestingT(t) }

type LinkSuite struct{}

var _ = Suite(&LinkSuite{})

// TODO: add more tests
var linkParseTests = []struct {
	in  string
	out []Link
}{
	{
		"<http://example.com/TheBook/chapter2>; rel=\"previous\";\n    title=\"previous chapter\"",
		[]Link{{URI: "http://example.com/TheBook/chapter2", Rel: "previous", Params: map[string]string{"title": "previous chapter"}}},
	},
	{
		"</TheBook/chapter2>;\n rel=\"previous\"; title*=UTF-8'de'letztes%20Kapitel,\n </TheBook/chapter4>;\n rel=\"next\"; title*=UTF-8'de'n%c3%a4chstes%20Kapitel",
		[]Link{
			{URI: "/TheBook/chapter2", Rel: "previous", Params: map[string]string{"title*": "UTF-8'de'letztes%20Kapitel"}},
			{URI: "/TheBook/chapter4", Rel: "next", Params: map[string]string{"title*": "UTF-8'de'n%c3%a4chstes%20Kapitel"}},
		},
	},
}

func (s *LinkSuite) TestLinkParsing(c *C) {
	for i, t := range linkParseTests {
		res, err := Parse(t.in)
		c.Assert(err, IsNil, Commentf("test %d", i))
		c.Assert(res, DeepEquals, t.out, Commentf("test %d", i))
	}
}

var linkFormatTests = []struct {
	in  []Link
	out string
}{
	{
		[]Link{{URI: "/a", Rel: "foo", Params: map[string]string{"a": "b", "c": "d"}}},
		`</a>; rel="foo"; a="b"; c="d"`,
	},
	{
		[]Link{
			{URI: "/b", Rel: "foo", Params: map[string]string{"a": "b", "c": "d"}},
			{URI: "/a", Rel: "foo", Params: map[string]string{"a": "b", "c": "d"}}},
		`</b>; rel="foo"; a="b"; c="d", </a>; rel="foo"; a="b"; c="d"`,
	},
}

func (s *LinkSuite) TestLinkGeneration(c *C) {
	for i, t := range linkFormatTests {
		res := Format(t.in)
		cm := Commentf("test %d", i)
		c.Assert(res, Equals, t.out, cm)
		parsed, err := Parse(res)
		c.Assert(err, IsNil, cm)
		c.Assert(parsed, DeepEquals, t.in, cm)
	}
}
