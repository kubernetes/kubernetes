// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"strconv"
	"strings"
)

type TokenSourceCharStreamPair struct {
	tokenSource TokenSource
	charStream  CharStream
}

// A token has properties: text, type, line, character position in the line
// (so we can ignore tabs), token channel, index, and source from which
// we obtained this token.

type Token interface {
	GetSource() *TokenSourceCharStreamPair
	GetTokenType() int
	GetChannel() int
	GetStart() int
	GetStop() int
	GetLine() int
	GetColumn() int

	GetText() string
	SetText(s string)

	GetTokenIndex() int
	SetTokenIndex(v int)

	GetTokenSource() TokenSource
	GetInputStream() CharStream
}

type BaseToken struct {
	source     *TokenSourceCharStreamPair
	tokenType  int    // token type of the token
	channel    int    // The parser ignores everything not on DEFAULT_CHANNEL
	start      int    // optional return -1 if not implemented.
	stop       int    // optional return -1 if not implemented.
	tokenIndex int    // from 0..n-1 of the token object in the input stream
	line       int    // line=1..n of the 1st character
	column     int    // beginning of the line at which it occurs, 0..n-1
	text       string // text of the token.
	readOnly   bool
}

const (
	TokenInvalidType = 0

	// During lookahead operations, this "token" signifies we hit rule end ATN state
	// and did not follow it despite needing to.
	TokenEpsilon = -2

	TokenMinUserTokenType = 1

	TokenEOF = -1

	// All tokens go to the parser (unless Skip() is called in that rule)
	// on a particular "channel". The parser tunes to a particular channel
	// so that whitespace etc... can go to the parser on a "hidden" channel.

	TokenDefaultChannel = 0

	// Anything on different channel than DEFAULT_CHANNEL is not parsed
	// by parser.

	TokenHiddenChannel = 1
)

func (b *BaseToken) GetChannel() int {
	return b.channel
}

func (b *BaseToken) GetStart() int {
	return b.start
}

func (b *BaseToken) GetStop() int {
	return b.stop
}

func (b *BaseToken) GetLine() int {
	return b.line
}

func (b *BaseToken) GetColumn() int {
	return b.column
}

func (b *BaseToken) GetTokenType() int {
	return b.tokenType
}

func (b *BaseToken) GetSource() *TokenSourceCharStreamPair {
	return b.source
}

func (b *BaseToken) GetTokenIndex() int {
	return b.tokenIndex
}

func (b *BaseToken) SetTokenIndex(v int) {
	b.tokenIndex = v
}

func (b *BaseToken) GetTokenSource() TokenSource {
	return b.source.tokenSource
}

func (b *BaseToken) GetInputStream() CharStream {
	return b.source.charStream
}

type CommonToken struct {
	*BaseToken
}

func NewCommonToken(source *TokenSourceCharStreamPair, tokenType, channel, start, stop int) *CommonToken {

	t := new(CommonToken)

	t.BaseToken = new(BaseToken)

	t.source = source
	t.tokenType = tokenType
	t.channel = channel
	t.start = start
	t.stop = stop
	t.tokenIndex = -1
	if t.source.tokenSource != nil {
		t.line = source.tokenSource.GetLine()
		t.column = source.tokenSource.GetCharPositionInLine()
	} else {
		t.column = -1
	}
	return t
}

// An empty {@link Pair} which is used as the default value of
// {@link //source} for tokens that do not have a source.

//CommonToken.EMPTY_SOURCE = [ nil, nil ]

// Constructs a New{@link CommonToken} as a copy of another {@link Token}.
//
// <p>
// If {@code oldToken} is also a {@link CommonToken} instance, the newly
// constructed token will share a reference to the {@link //text} field and
// the {@link Pair} stored in {@link //source}. Otherwise, {@link //text} will
// be assigned the result of calling {@link //GetText}, and {@link //source}
// will be constructed from the result of {@link Token//GetTokenSource} and
// {@link Token//GetInputStream}.</p>
//
// @param oldToken The token to copy.
func (c *CommonToken) clone() *CommonToken {
	t := NewCommonToken(c.source, c.tokenType, c.channel, c.start, c.stop)
	t.tokenIndex = c.GetTokenIndex()
	t.line = c.GetLine()
	t.column = c.GetColumn()
	t.text = c.GetText()
	return t
}

func (c *CommonToken) GetText() string {
	if c.text != "" {
		return c.text
	}
	input := c.GetInputStream()
	if input == nil {
		return ""
	}
	n := input.Size()
	if c.start < n && c.stop < n {
		return input.GetTextFromInterval(NewInterval(c.start, c.stop))
	}
	return "<EOF>"
}

func (c *CommonToken) SetText(text string) {
	c.text = text
}

func (c *CommonToken) String() string {
	txt := c.GetText()
	if txt != "" {
		txt = strings.Replace(txt, "\n", "\\n", -1)
		txt = strings.Replace(txt, "\r", "\\r", -1)
		txt = strings.Replace(txt, "\t", "\\t", -1)
	} else {
		txt = "<no text>"
	}

	var ch string
	if c.channel > 0 {
		ch = ",channel=" + strconv.Itoa(c.channel)
	} else {
		ch = ""
	}

	return "[@" + strconv.Itoa(c.tokenIndex) + "," + strconv.Itoa(c.start) + ":" + strconv.Itoa(c.stop) + "='" +
		txt + "',<" + strconv.Itoa(c.tokenType) + ">" +
		ch + "," + strconv.Itoa(c.line) + ":" + strconv.Itoa(c.column) + "]"
}
