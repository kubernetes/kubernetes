// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"strconv"
)

// CommonTokenStream is an implementation of TokenStream that loads tokens from
// a TokenSource on-demand and places the tokens in a buffer to provide access
// to any previous token by index. This token stream ignores the value of
// Token.getChannel. If your parser requires the token stream filter tokens to
// only those on a particular channel, such as Token.DEFAULT_CHANNEL or
// Token.HIDDEN_CHANNEL, use a filtering token stream such a CommonTokenStream.
type CommonTokenStream struct {
	channel int

	// fetchedEOF indicates whether the Token.EOF token has been fetched from
	// tokenSource and added to tokens. This field improves performance for the
	// following cases:
	//
	// consume: The lookahead check in consume to preven consuming the EOF symbol is
	// optimized by checking the values of fetchedEOF and p instead of calling LA.
	//
	// fetch: The check to prevent adding multiple EOF symbols into tokens is
	// trivial with bt field.
	fetchedEOF bool

	// index indexs into tokens of the current token (next token to consume).
	// tokens[p] should be LT(1). It is set to -1 when the stream is first
	// constructed or when SetTokenSource is called, indicating that the first token
	// has not yet been fetched from the token source. For additional information,
	// see the documentation of IntStream for a description of initializing methods.
	index int

	// tokenSource is the TokenSource from which tokens for the bt stream are
	// fetched.
	tokenSource TokenSource

	// tokens is all tokens fetched from the token source. The list is considered a
	// complete view of the input once fetchedEOF is set to true.
	tokens []Token
}

func NewCommonTokenStream(lexer Lexer, channel int) *CommonTokenStream {
	return &CommonTokenStream{
		channel:     channel,
		index:       -1,
		tokenSource: lexer,
		tokens:      make([]Token, 0),
	}
}

func (c *CommonTokenStream) GetAllTokens() []Token {
	return c.tokens
}

func (c *CommonTokenStream) Mark() int {
	return 0
}

func (c *CommonTokenStream) Release(marker int) {}

func (c *CommonTokenStream) reset() {
	c.Seek(0)
}

func (c *CommonTokenStream) Seek(index int) {
	c.lazyInit()
	c.index = c.adjustSeekIndex(index)
}

func (c *CommonTokenStream) Get(index int) Token {
	c.lazyInit()

	return c.tokens[index]
}

func (c *CommonTokenStream) Consume() {
	SkipEOFCheck := false

	if c.index >= 0 {
		if c.fetchedEOF {
			// The last token in tokens is EOF. Skip the check if p indexes any fetched.
			// token except the last.
			SkipEOFCheck = c.index < len(c.tokens)-1
		} else {
			// No EOF token in tokens. Skip the check if p indexes a fetched token.
			SkipEOFCheck = c.index < len(c.tokens)
		}
	} else {
		// Not yet initialized
		SkipEOFCheck = false
	}

	if !SkipEOFCheck && c.LA(1) == TokenEOF {
		panic("cannot consume EOF")
	}

	if c.Sync(c.index + 1) {
		c.index = c.adjustSeekIndex(c.index + 1)
	}
}

// Sync makes sure index i in tokens has a token and returns true if a token is
// located at index i and otherwise false.
func (c *CommonTokenStream) Sync(i int) bool {
	n := i - len(c.tokens) + 1 // TODO: How many more elements do we need?

	if n > 0 {
		fetched := c.fetch(n)
		return fetched >= n
	}

	return true
}

// fetch adds n elements to buffer and returns the actual number of elements
// added to the buffer.
func (c *CommonTokenStream) fetch(n int) int {
	if c.fetchedEOF {
		return 0
	}

	for i := 0; i < n; i++ {
		t := c.tokenSource.NextToken()

		t.SetTokenIndex(len(c.tokens))
		c.tokens = append(c.tokens, t)

		if t.GetTokenType() == TokenEOF {
			c.fetchedEOF = true

			return i + 1
		}
	}

	return n
}

// GetTokens gets all tokens from start to stop inclusive.
func (c *CommonTokenStream) GetTokens(start int, stop int, types *IntervalSet) []Token {
	if start < 0 || stop < 0 {
		return nil
	}

	c.lazyInit()

	subset := make([]Token, 0)

	if stop >= len(c.tokens) {
		stop = len(c.tokens) - 1
	}

	for i := start; i < stop; i++ {
		t := c.tokens[i]

		if t.GetTokenType() == TokenEOF {
			break
		}

		if types == nil || types.contains(t.GetTokenType()) {
			subset = append(subset, t)
		}
	}

	return subset
}

func (c *CommonTokenStream) LA(i int) int {
	return c.LT(i).GetTokenType()
}

func (c *CommonTokenStream) lazyInit() {
	if c.index == -1 {
		c.setup()
	}
}

func (c *CommonTokenStream) setup() {
	c.Sync(0)
	c.index = c.adjustSeekIndex(0)
}

func (c *CommonTokenStream) GetTokenSource() TokenSource {
	return c.tokenSource
}

// SetTokenSource resets the c token stream by setting its token source.
func (c *CommonTokenStream) SetTokenSource(tokenSource TokenSource) {
	c.tokenSource = tokenSource
	c.tokens = make([]Token, 0)
	c.index = -1
}

// NextTokenOnChannel returns the index of the next token on channel given a
// starting index. Returns i if tokens[i] is on channel. Returns -1 if there are
// no tokens on channel between i and EOF.
func (c *CommonTokenStream) NextTokenOnChannel(i, channel int) int {
	c.Sync(i)

	if i >= len(c.tokens) {
		return -1
	}

	token := c.tokens[i]

	for token.GetChannel() != c.channel {
		if token.GetTokenType() == TokenEOF {
			return -1
		}

		i++
		c.Sync(i)
		token = c.tokens[i]
	}

	return i
}

// previousTokenOnChannel returns the index of the previous token on channel
// given a starting index. Returns i if tokens[i] is on channel. Returns -1 if
// there are no tokens on channel between i and 0.
func (c *CommonTokenStream) previousTokenOnChannel(i, channel int) int {
	for i >= 0 && c.tokens[i].GetChannel() != channel {
		i--
	}

	return i
}

// GetHiddenTokensToRight collects all tokens on a specified channel to the
// right of the current token up until we see a token on DEFAULT_TOKEN_CHANNEL
// or EOF. If channel is -1, it finds any non-default channel token.
func (c *CommonTokenStream) GetHiddenTokensToRight(tokenIndex, channel int) []Token {
	c.lazyInit()

	if tokenIndex < 0 || tokenIndex >= len(c.tokens) {
		panic(strconv.Itoa(tokenIndex) + " not in 0.." + strconv.Itoa(len(c.tokens)-1))
	}

	nextOnChannel := c.NextTokenOnChannel(tokenIndex+1, LexerDefaultTokenChannel)
	from := tokenIndex + 1

	// If no onchannel to the right, then nextOnChannel == -1, so set to to last token
	var to int

	if nextOnChannel == -1 {
		to = len(c.tokens) - 1
	} else {
		to = nextOnChannel
	}

	return c.filterForChannel(from, to, channel)
}

// GetHiddenTokensToLeft collects all tokens on channel to the left of the
// current token until we see a token on DEFAULT_TOKEN_CHANNEL. If channel is
// -1, it finds any non default channel token.
func (c *CommonTokenStream) GetHiddenTokensToLeft(tokenIndex, channel int) []Token {
	c.lazyInit()

	if tokenIndex < 0 || tokenIndex >= len(c.tokens) {
		panic(strconv.Itoa(tokenIndex) + " not in 0.." + strconv.Itoa(len(c.tokens)-1))
	}

	prevOnChannel := c.previousTokenOnChannel(tokenIndex-1, LexerDefaultTokenChannel)

	if prevOnChannel == tokenIndex-1 {
		return nil
	}

	// If there are none on channel to the left and prevOnChannel == -1 then from = 0
	from := prevOnChannel + 1
	to := tokenIndex - 1

	return c.filterForChannel(from, to, channel)
}

func (c *CommonTokenStream) filterForChannel(left, right, channel int) []Token {
	hidden := make([]Token, 0)

	for i := left; i < right+1; i++ {
		t := c.tokens[i]

		if channel == -1 {
			if t.GetChannel() != LexerDefaultTokenChannel {
				hidden = append(hidden, t)
			}
		} else if t.GetChannel() == channel {
			hidden = append(hidden, t)
		}
	}

	if len(hidden) == 0 {
		return nil
	}

	return hidden
}

func (c *CommonTokenStream) GetSourceName() string {
	return c.tokenSource.GetSourceName()
}

func (c *CommonTokenStream) Size() int {
	return len(c.tokens)
}

func (c *CommonTokenStream) Index() int {
	return c.index
}

func (c *CommonTokenStream) GetAllText() string {
	return c.GetTextFromInterval(nil)
}

func (c *CommonTokenStream) GetTextFromTokens(start, end Token) string {
	if start == nil || end == nil {
		return ""
	}

	return c.GetTextFromInterval(NewInterval(start.GetTokenIndex(), end.GetTokenIndex()))
}

func (c *CommonTokenStream) GetTextFromRuleContext(interval RuleContext) string {
	return c.GetTextFromInterval(interval.GetSourceInterval())
}

func (c *CommonTokenStream) GetTextFromInterval(interval *Interval) string {
	c.lazyInit()

	if interval == nil {
		c.Fill()
		interval = NewInterval(0, len(c.tokens)-1)
	} else {
		c.Sync(interval.Stop)
	}

	start := interval.Start
	stop := interval.Stop

	if start < 0 || stop < 0 {
		return ""
	}

	if stop >= len(c.tokens) {
		stop = len(c.tokens) - 1
	}

	s := ""

	for i := start; i < stop+1; i++ {
		t := c.tokens[i]

		if t.GetTokenType() == TokenEOF {
			break
		}

		s += t.GetText()
	}

	return s
}

// Fill gets all tokens from the lexer until EOF.
func (c *CommonTokenStream) Fill() {
	c.lazyInit()

	for c.fetch(1000) == 1000 {
		continue
	}
}

func (c *CommonTokenStream) adjustSeekIndex(i int) int {
	return c.NextTokenOnChannel(i, c.channel)
}

func (c *CommonTokenStream) LB(k int) Token {
	if k == 0 || c.index-k < 0 {
		return nil
	}

	i := c.index
	n := 1

	// Find k good tokens looking backward
	for n <= k {
		// Skip off-channel tokens
		i = c.previousTokenOnChannel(i-1, c.channel)
		n++
	}

	if i < 0 {
		return nil
	}

	return c.tokens[i]
}

func (c *CommonTokenStream) LT(k int) Token {
	c.lazyInit()

	if k == 0 {
		return nil
	}

	if k < 0 {
		return c.LB(-k)
	}

	i := c.index
	n := 1 // We know tokens[n] is valid

	// Find k good tokens
	for n < k {
		// Skip off-channel tokens, but make sure to not look past EOF
		if c.Sync(i + 1) {
			i = c.NextTokenOnChannel(i+1, c.channel)
		}

		n++
	}

	return c.tokens[i]
}

// getNumberOfOnChannelTokens counts EOF once.
func (c *CommonTokenStream) getNumberOfOnChannelTokens() int {
	var n int

	c.Fill()

	for i := 0; i < len(c.tokens); i++ {
		t := c.tokens[i]

		if t.GetChannel() == c.channel {
			n++
		}

		if t.GetTokenType() == TokenEOF {
			break
		}
	}

	return n
}
