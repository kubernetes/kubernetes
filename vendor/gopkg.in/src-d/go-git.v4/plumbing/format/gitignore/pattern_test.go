package gitignore

import (
	"testing"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type PatternSuite struct{}

var _ = Suite(&PatternSuite{})

func (s *PatternSuite) TestSimpleMatch_inclusion(c *C) {
	p := ParsePattern("!vul?ano", nil)
	r := p.Match([]string{"value", "vulkano", "tail"}, false)
	c.Assert(r, Equals, Include)
}

func (s *PatternSuite) TestMatch_domainLonger_mismatch(c *C) {
	p := ParsePattern("value", []string{"head", "middle", "tail"})
	r := p.Match([]string{"head", "middle"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestMatch_domainSameLength_mismatch(c *C) {
	p := ParsePattern("value", []string{"head", "middle", "tail"})
	r := p.Match([]string{"head", "middle", "tail"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestMatch_domainMismatch_mismatch(c *C) {
	p := ParsePattern("value", []string{"head", "middle", "tail"})
	r := p.Match([]string{"head", "middle", "_tail_", "value"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestSimpleMatch_withDomain(c *C) {
	p := ParsePattern("middle/", []string{"value", "volcano"})
	r := p.Match([]string{"value", "volcano", "middle", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestSimpleMatch_onlyMatchInDomain_mismatch(c *C) {
	p := ParsePattern("volcano/", []string{"value", "volcano"})
	r := p.Match([]string{"value", "volcano", "tail"}, true)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestSimpleMatch_atStart(c *C) {
	p := ParsePattern("value", nil)
	r := p.Match([]string{"value", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestSimpleMatch_inTheMiddle(c *C) {
	p := ParsePattern("value", nil)
	r := p.Match([]string{"head", "value", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestSimpleMatch_atEnd(c *C) {
	p := ParsePattern("value", nil)
	r := p.Match([]string{"head", "value"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestSimpleMatch_atStart_dirWanted(c *C) {
	p := ParsePattern("value/", nil)
	r := p.Match([]string{"value", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestSimpleMatch_inTheMiddle_dirWanted(c *C) {
	p := ParsePattern("value/", nil)
	r := p.Match([]string{"head", "value", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestSimpleMatch_atEnd_dirWanted(c *C) {
	p := ParsePattern("value/", nil)
	r := p.Match([]string{"head", "value"}, true)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestSimpleMatch_atEnd_dirWanted_notADir_mismatch(c *C) {
	p := ParsePattern("value/", nil)
	r := p.Match([]string{"head", "value"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestSimpleMatch_mismatch(c *C) {
	p := ParsePattern("value", nil)
	r := p.Match([]string{"head", "val", "tail"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestSimpleMatch_valueLonger_mismatch(c *C) {
	p := ParsePattern("val", nil)
	r := p.Match([]string{"head", "value", "tail"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestSimpleMatch_withAsterisk(c *C) {
	p := ParsePattern("v*o", nil)
	r := p.Match([]string{"value", "vulkano", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestSimpleMatch_withQuestionMark(c *C) {
	p := ParsePattern("vul?ano", nil)
	r := p.Match([]string{"value", "vulkano", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestSimpleMatch_magicChars(c *C) {
	p := ParsePattern("v[ou]l[kc]ano", nil)
	r := p.Match([]string{"value", "volcano"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestSimpleMatch_wrongPattern_mismatch(c *C) {
	p := ParsePattern("v[ou]l[", nil)
	r := p.Match([]string{"value", "vol["}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestGlobMatch_fromRootWithSlash(c *C) {
	p := ParsePattern("/value/vul?ano", nil)
	r := p.Match([]string{"value", "vulkano", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_withDomain(c *C) {
	p := ParsePattern("middle/tail/", []string{"value", "volcano"})
	r := p.Match([]string{"value", "volcano", "middle", "tail"}, true)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_onlyMatchInDomain_mismatch(c *C) {
	p := ParsePattern("volcano/tail", []string{"value", "volcano"})
	r := p.Match([]string{"value", "volcano", "tail"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestGlobMatch_fromRootWithoutSlash(c *C) {
	p := ParsePattern("value/vul?ano", nil)
	r := p.Match([]string{"value", "vulkano", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_fromRoot_mismatch(c *C) {
	p := ParsePattern("value/vulkano", nil)
	r := p.Match([]string{"value", "volcano"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestGlobMatch_fromRoot_tooShort_mismatch(c *C) {
	p := ParsePattern("value/vul?ano", nil)
	r := p.Match([]string{"value"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestGlobMatch_fromRoot_notAtRoot_mismatch(c *C) {
	p := ParsePattern("/value/volcano", nil)
	r := p.Match([]string{"value", "value", "volcano"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestGlobMatch_leadingAsterisks_atStart(c *C) {
	p := ParsePattern("**/*lue/vol?ano", nil)
	r := p.Match([]string{"value", "volcano", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_leadingAsterisks_notAtStart(c *C) {
	p := ParsePattern("**/*lue/vol?ano", nil)
	r := p.Match([]string{"head", "value", "volcano", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_leadingAsterisks_mismatch(c *C) {
	p := ParsePattern("**/*lue/vol?ano", nil)
	r := p.Match([]string{"head", "value", "Volcano", "tail"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestGlobMatch_leadingAsterisks_isDir(c *C) {
	p := ParsePattern("**/*lue/vol?ano/", nil)
	r := p.Match([]string{"head", "value", "volcano", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_leadingAsterisks_isDirAtEnd(c *C) {
	p := ParsePattern("**/*lue/vol?ano/", nil)
	r := p.Match([]string{"head", "value", "volcano"}, true)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_leadingAsterisks_isDir_mismatch(c *C) {
	p := ParsePattern("**/*lue/vol?ano/", nil)
	r := p.Match([]string{"head", "value", "Colcano"}, true)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestGlobMatch_leadingAsterisks_isDirNoDirAtEnd_mismatch(c *C) {
	p := ParsePattern("**/*lue/vol?ano/", nil)
	r := p.Match([]string{"head", "value", "volcano"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestGlobMatch_tailingAsterisks(c *C) {
	p := ParsePattern("/*lue/vol?ano/**", nil)
	r := p.Match([]string{"value", "volcano", "tail", "moretail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_tailingAsterisks_exactMatch(c *C) {
	p := ParsePattern("/*lue/vol?ano/**", nil)
	r := p.Match([]string{"value", "volcano"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_middleAsterisks_emptyMatch(c *C) {
	p := ParsePattern("/*lue/**/vol?ano", nil)
	r := p.Match([]string{"value", "volcano"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_middleAsterisks_oneMatch(c *C) {
	p := ParsePattern("/*lue/**/vol?ano", nil)
	r := p.Match([]string{"value", "middle", "volcano"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_middleAsterisks_multiMatch(c *C) {
	p := ParsePattern("/*lue/**/vol?ano", nil)
	r := p.Match([]string{"value", "middle1", "middle2", "volcano"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_middleAsterisks_isDir_trailing(c *C) {
	p := ParsePattern("/*lue/**/vol?ano/", nil)
	r := p.Match([]string{"value", "middle1", "middle2", "volcano"}, true)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_middleAsterisks_isDir_trailing_mismatch(c *C) {
	p := ParsePattern("/*lue/**/vol?ano/", nil)
	r := p.Match([]string{"value", "middle1", "middle2", "volcano"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestGlobMatch_middleAsterisks_isDir(c *C) {
	p := ParsePattern("/*lue/**/vol?ano/", nil)
	r := p.Match([]string{"value", "middle1", "middle2", "volcano", "tail"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_wrongDoubleAsterisk_mismatch(c *C) {
	p := ParsePattern("/*lue/**foo/vol?ano", nil)
	r := p.Match([]string{"value", "foo", "volcano", "tail"}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestGlobMatch_magicChars(c *C) {
	p := ParsePattern("**/head/v[ou]l[kc]ano", nil)
	r := p.Match([]string{"value", "head", "volcano"}, false)
	c.Assert(r, Equals, Exclude)
}

func (s *PatternSuite) TestGlobMatch_wrongPattern_noTraversal_mismatch(c *C) {
	p := ParsePattern("**/head/v[ou]l[", nil)
	r := p.Match([]string{"value", "head", "vol["}, false)
	c.Assert(r, Equals, NoMatch)
}

func (s *PatternSuite) TestGlobMatch_wrongPattern_onTraversal_mismatch(c *C) {
	p := ParsePattern("/value/**/v[ou]l[", nil)
	r := p.Match([]string{"value", "head", "vol["}, false)
	c.Assert(r, Equals, NoMatch)
}
