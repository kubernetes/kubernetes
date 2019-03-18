package ini

// skipper is used to skip certain blocks of an ini file.
// Currently skipper is used to skip nested blocks of ini
// files. See example below
//
//	[ foo ]
//	nested = ; this section will be skipped
//		a=b
//		c=d
//	bar=baz ; this will be included
type skipper struct {
	shouldSkip bool
	TokenSet   bool
	prevTok    Token
}

func newSkipper() skipper {
	return skipper{
		prevTok: emptyToken,
	}
}

func (s *skipper) ShouldSkip(tok Token) bool {
	if s.shouldSkip &&
		s.prevTok.Type() == TokenNL &&
		tok.Type() != TokenWS {

		s.Continue()
		return false
	}
	s.prevTok = tok

	return s.shouldSkip
}

func (s *skipper) Skip() {
	s.shouldSkip = true
	s.prevTok = emptyToken
}

func (s *skipper) Continue() {
	s.shouldSkip = false
	s.prevTok = emptyToken
}
