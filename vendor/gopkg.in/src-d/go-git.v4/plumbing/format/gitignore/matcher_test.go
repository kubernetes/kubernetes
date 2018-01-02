package gitignore

import (
	. "gopkg.in/check.v1"
)

func (s *MatcherSuite) TestMatcher_Match(c *C) {
	ps := []Pattern{
		ParsePattern("**/middle/v[uo]l?ano", nil),
		ParsePattern("!volcano", nil),
	}

	m := NewMatcher(ps)
	c.Assert(m.Match([]string{"head", "middle", "vulkano"}, false), Equals, true)
	c.Assert(m.Match([]string{"head", "middle", "volcano"}, false), Equals, false)
}
