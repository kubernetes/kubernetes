package packp

import (
	"bytes"
	"io"
	"strings"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability"

	. "gopkg.in/check.v1"
)

type AdvRefsDecodeSuite struct{}

var _ = Suite(&AdvRefsDecodeSuite{})

func (s *AdvRefsDecodeSuite) TestEmpty(c *C) {
	var buf bytes.Buffer
	ar := NewAdvRefs()
	c.Assert(ar.Decode(&buf), Equals, ErrEmptyInput)
}

func (s *AdvRefsDecodeSuite) TestEmptyFlush(c *C) {
	var buf bytes.Buffer
	e := pktline.NewEncoder(&buf)
	e.Flush()
	ar := NewAdvRefs()
	c.Assert(ar.Decode(&buf), Equals, ErrEmptyAdvRefs)
}

func (s *AdvRefsDecodeSuite) TestEmptyPrefixFlush(c *C) {
	var buf bytes.Buffer
	e := pktline.NewEncoder(&buf)
	e.EncodeString("# service=git-upload-pack")
	e.Flush()
	e.Flush()
	ar := NewAdvRefs()
	c.Assert(ar.Decode(&buf), Equals, ErrEmptyAdvRefs)
}

func (s *AdvRefsDecodeSuite) TestShortForHash(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*too short.*")
}

func (s *AdvRefsDecodeSuite) testDecoderErrorMatches(c *C, input io.Reader, pattern string) {
	ar := NewAdvRefs()
	c.Assert(ar.Decode(input), ErrorMatches, pattern)
}

func (s *AdvRefsDecodeSuite) TestInvalidFirstHash(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796alberto2219af86ec6584e5 HEAD\x00multi_ack thin-pack\n",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*invalid hash.*")
}

func (s *AdvRefsDecodeSuite) TestZeroId(c *C) {
	payloads := []string{
		"0000000000000000000000000000000000000000 capabilities^{}\x00multi_ack thin-pack\n",
		pktline.FlushString,
	}
	ar := s.testDecodeOK(c, payloads)
	c.Assert(ar.Head, IsNil)
}

func (s *AdvRefsDecodeSuite) testDecodeOK(c *C, payloads []string) *AdvRefs {
	var buf bytes.Buffer
	e := pktline.NewEncoder(&buf)
	err := e.EncodeString(payloads...)
	c.Assert(err, IsNil)

	ar := NewAdvRefs()
	c.Assert(ar.Decode(&buf), IsNil)

	return ar
}

func (s *AdvRefsDecodeSuite) TestMalformedZeroId(c *C) {
	payloads := []string{
		"0000000000000000000000000000000000000000 wrong\x00multi_ack thin-pack\n",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*malformed zero-id.*")
}

func (s *AdvRefsDecodeSuite) TestShortZeroId(c *C) {
	payloads := []string{
		"0000000000000000000000000000000000000000 capabi",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*too short zero-id.*")
}

func (s *AdvRefsDecodeSuite) TestHead(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00",
		pktline.FlushString,
	}
	ar := s.testDecodeOK(c, payloads)
	c.Assert(*ar.Head, Equals,
		plumbing.NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5"))
}

func (s *AdvRefsDecodeSuite) TestFirstIsNotHead(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 refs/heads/master\x00",
		pktline.FlushString,
	}
	ar := s.testDecodeOK(c, payloads)
	c.Assert(ar.Head, IsNil)
	c.Assert(ar.References["refs/heads/master"], Equals,
		plumbing.NewHash("6ecf0ef2c2dffb796033e5a02219af86ec6584e5"))
}

func (s *AdvRefsDecodeSuite) TestShortRef(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 H",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*too short.*")
}

func (s *AdvRefsDecodeSuite) TestNoNULL(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEADofs-delta multi_ack",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*NULL not found.*")
}

func (s *AdvRefsDecodeSuite) TestNoSpaceAfterHash(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5-HEAD\x00",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*no space after hash.*")
}

func (s *AdvRefsDecodeSuite) TestNoCaps(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00",
		pktline.FlushString,
	}
	ar := s.testDecodeOK(c, payloads)
	c.Assert(ar.Capabilities.IsEmpty(), Equals, true)
}

func (s *AdvRefsDecodeSuite) TestCaps(c *C) {
	type entry struct {
		Name   capability.Capability
		Values []string
	}

	for _, test := range [...]struct {
		input        []string
		capabilities []entry
	}{{
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00",
			pktline.FlushString,
		},
		capabilities: []entry{},
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00\n",
			pktline.FlushString,
		},
		capabilities: []entry{},
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta",
			pktline.FlushString,
		},
		capabilities: []entry{
			{
				Name:   capability.OFSDelta,
				Values: []string(nil),
			},
		},
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta multi_ack",
			pktline.FlushString,
		},
		capabilities: []entry{
			{Name: capability.OFSDelta, Values: []string(nil)},
			{Name: capability.MultiACK, Values: []string(nil)},
		},
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta multi_ack\n",
			pktline.FlushString,
		},
		capabilities: []entry{
			{Name: capability.OFSDelta, Values: []string(nil)},
			{Name: capability.MultiACK, Values: []string(nil)},
		},
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00symref=HEAD:refs/heads/master agent=foo=bar\n",
			pktline.FlushString,
		},
		capabilities: []entry{
			{Name: capability.SymRef, Values: []string{"HEAD:refs/heads/master"}},
			{Name: capability.Agent, Values: []string{"foo=bar"}},
		},
	}} {
		ar := s.testDecodeOK(c, test.input)
		for _, fixCap := range test.capabilities {
			c.Assert(ar.Capabilities.Supports(fixCap.Name), Equals, true,
				Commentf("input = %q, capability = %q", test.input, fixCap.Name))
			c.Assert(ar.Capabilities.Get(fixCap.Name), DeepEquals, fixCap.Values,
				Commentf("input = %q, capability = %q", test.input, fixCap.Name))
		}
	}
}

func (s *AdvRefsDecodeSuite) TestWithPrefix(c *C) {
	payloads := []string{
		"# this is a prefix\n",
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta\n",
		pktline.FlushString,
	}
	ar := s.testDecodeOK(c, payloads)
	c.Assert(len(ar.Prefix), Equals, 1)
	c.Assert(ar.Prefix[0], DeepEquals, []byte("# this is a prefix"))
}

func (s *AdvRefsDecodeSuite) TestWithPrefixAndFlush(c *C) {
	payloads := []string{
		"# this is a prefix\n",
		pktline.FlushString,
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta\n",
		pktline.FlushString,
	}
	ar := s.testDecodeOK(c, payloads)
	c.Assert(len(ar.Prefix), Equals, 2)
	c.Assert(ar.Prefix[0], DeepEquals, []byte("# this is a prefix"))
	c.Assert(ar.Prefix[1], DeepEquals, []byte(pktline.FlushString))
}

func (s *AdvRefsDecodeSuite) TestOtherRefs(c *C) {
	for _, test := range [...]struct {
		input      []string
		references map[string]plumbing.Hash
		peeled     map[string]plumbing.Hash
	}{{
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
			pktline.FlushString,
		},
		references: make(map[string]plumbing.Hash),
		peeled:     make(map[string]plumbing.Hash),
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
			"1111111111111111111111111111111111111111 ref/foo",
			pktline.FlushString,
		},
		references: map[string]plumbing.Hash{
			"ref/foo": plumbing.NewHash("1111111111111111111111111111111111111111"),
		},
		peeled: make(map[string]plumbing.Hash),
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
			"1111111111111111111111111111111111111111 ref/foo\n",
			pktline.FlushString,
		},
		references: map[string]plumbing.Hash{
			"ref/foo": plumbing.NewHash("1111111111111111111111111111111111111111"),
		},
		peeled: make(map[string]plumbing.Hash),
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
			"1111111111111111111111111111111111111111 ref/foo\n",
			"2222222222222222222222222222222222222222 ref/bar",
			pktline.FlushString,
		},
		references: map[string]plumbing.Hash{
			"ref/foo": plumbing.NewHash("1111111111111111111111111111111111111111"),
			"ref/bar": plumbing.NewHash("2222222222222222222222222222222222222222"),
		},
		peeled: make(map[string]plumbing.Hash),
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
			"1111111111111111111111111111111111111111 ref/foo^{}\n",
			pktline.FlushString,
		},
		references: make(map[string]plumbing.Hash),
		peeled: map[string]plumbing.Hash{
			"ref/foo": plumbing.NewHash("1111111111111111111111111111111111111111"),
		},
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
			"1111111111111111111111111111111111111111 ref/foo\n",
			"2222222222222222222222222222222222222222 ref/bar^{}",
			pktline.FlushString,
		},
		references: map[string]plumbing.Hash{
			"ref/foo": plumbing.NewHash("1111111111111111111111111111111111111111"),
		},
		peeled: map[string]plumbing.Hash{
			"ref/bar": plumbing.NewHash("2222222222222222222222222222222222222222"),
		},
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
			"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
			"51b8b4fb32271d39fbdd760397406177b2b0fd36 refs/pull/10/head\n",
			"02b5a6031ba7a8cbfde5d65ff9e13ecdbc4a92ca refs/pull/100/head\n",
			"c284c212704c43659bf5913656b8b28e32da1621 refs/pull/100/merge\n",
			"3d6537dce68c8b7874333a1720958bd8db3ae8ca refs/pull/101/merge\n",
			"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11\n",
			"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11^{}\n",
			"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
			"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
			pktline.FlushString,
		},
		references: map[string]plumbing.Hash{
			"refs/heads/master":      plumbing.NewHash("a6930aaee06755d1bdcfd943fbf614e4d92bb0c7"),
			"refs/pull/10/head":      plumbing.NewHash("51b8b4fb32271d39fbdd760397406177b2b0fd36"),
			"refs/pull/100/head":     plumbing.NewHash("02b5a6031ba7a8cbfde5d65ff9e13ecdbc4a92ca"),
			"refs/pull/100/merge":    plumbing.NewHash("c284c212704c43659bf5913656b8b28e32da1621"),
			"refs/pull/101/merge":    plumbing.NewHash("3d6537dce68c8b7874333a1720958bd8db3ae8ca"),
			"refs/tags/v2.6.11":      plumbing.NewHash("5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c"),
			"refs/tags/v2.6.11-tree": plumbing.NewHash("5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c"),
		},
		peeled: map[string]plumbing.Hash{
			"refs/tags/v2.6.11":      plumbing.NewHash("c39ae07f393806ccf406ef966e9a15afc43cc36a"),
			"refs/tags/v2.6.11-tree": plumbing.NewHash("c39ae07f393806ccf406ef966e9a15afc43cc36a"),
		},
	}} {
		ar := s.testDecodeOK(c, test.input)
		comment := Commentf("input = %v\n", test.input)
		c.Assert(ar.References, DeepEquals, test.references, comment)
		c.Assert(ar.Peeled, DeepEquals, test.peeled, comment)
	}
}

func (s *AdvRefsDecodeSuite) TestMalformedOtherRefsNoSpace(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00multi_ack thin-pack\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8crefs/tags/v2.6.11\n",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*malformed ref data.*")
}

func (s *AdvRefsDecodeSuite) TestMalformedOtherRefsMultipleSpaces(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00multi_ack thin-pack\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags v2.6.11\n",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*malformed ref data.*")
}

func (s *AdvRefsDecodeSuite) TestShallow(c *C) {
	for _, test := range [...]struct {
		input    []string
		shallows []plumbing.Hash
	}{{
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
			"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
			"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
			"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
			pktline.FlushString,
		},
		shallows: []plumbing.Hash{},
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
			"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
			"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
			"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
			"shallow 1111111111111111111111111111111111111111\n",
			pktline.FlushString,
		},
		shallows: []plumbing.Hash{plumbing.NewHash("1111111111111111111111111111111111111111")},
	}, {
		input: []string{
			"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
			"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
			"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
			"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
			"shallow 1111111111111111111111111111111111111111\n",
			"shallow 2222222222222222222222222222222222222222\n",
			pktline.FlushString,
		},
		shallows: []plumbing.Hash{
			plumbing.NewHash("1111111111111111111111111111111111111111"),
			plumbing.NewHash("2222222222222222222222222222222222222222"),
		},
	}} {
		ar := s.testDecodeOK(c, test.input)
		comment := Commentf("input = %v\n", test.input)
		c.Assert(ar.Shallows, DeepEquals, test.shallows, comment)
	}
}

func (s *AdvRefsDecodeSuite) TestInvalidShallowHash(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
		"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
		"shallow 11111111alcortes111111111111111111111111\n",
		"shallow 2222222222222222222222222222222222222222\n",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*invalid hash text.*")
}

func (s *AdvRefsDecodeSuite) TestGarbageAfterShallow(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
		"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
		"shallow 1111111111111111111111111111111111111111\n",
		"shallow 2222222222222222222222222222222222222222\n",
		"b5be40b90dbaa6bd337f3b77de361bfc0723468b refs/tags/v4.4",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*malformed shallow prefix.*")
}

func (s *AdvRefsDecodeSuite) TestMalformedShallowHash(c *C) {
	payloads := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
		"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
		"shallow 1111111111111111111111111111111111111111\n",
		"shallow 2222222222222222222222222222222222222222 malformed\n",
		pktline.FlushString,
	}
	r := toPktLines(c, payloads)
	s.testDecoderErrorMatches(c, r, ".*malformed shallow hash.*")
}

func (s *AdvRefsDecodeSuite) TestEOFRefs(c *C) {
	input := strings.NewReader("" +
		"005b6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n" +
		"003fa6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n" +
		"00355dc01c595e6c6ec9ccda4f6ffbf614e4d92bb0c7 refs/foo\n",
	)
	s.testDecoderErrorMatches(c, input, ".*invalid pkt-len.*")
}

func (s *AdvRefsDecodeSuite) TestEOFShallows(c *C) {
	input := strings.NewReader("" +
		"005b6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00ofs-delta symref=HEAD:/refs/heads/master\n" +
		"003fa6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n" +
		"00445dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n" +
		"0047c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n" +
		"0035shallow 1111111111111111111111111111111111111111\n" +
		"0034shallow 222222222222222222222222")
	s.testDecoderErrorMatches(c, input, ".*unexpected EOF.*")
}
