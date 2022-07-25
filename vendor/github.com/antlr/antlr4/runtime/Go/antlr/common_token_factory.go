// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

// TokenFactory creates CommonToken objects.
type TokenFactory interface {
	Create(source *TokenSourceCharStreamPair, ttype int, text string, channel, start, stop, line, column int) Token
}

// CommonTokenFactory is the default TokenFactory implementation.
type CommonTokenFactory struct {
	// copyText indicates whether CommonToken.setText should be called after
	// constructing tokens to explicitly set the text. This is useful for cases
	// where the input stream might not be able to provide arbitrary substrings of
	// text from the input after the lexer creates a token (e.g. the
	// implementation of CharStream.GetText in UnbufferedCharStream panics an
	// UnsupportedOperationException). Explicitly setting the token text allows
	// Token.GetText to be called at any time regardless of the input stream
	// implementation.
	//
	// The default value is false to avoid the performance and memory overhead of
	// copying text for every token unless explicitly requested.
	copyText bool
}

func NewCommonTokenFactory(copyText bool) *CommonTokenFactory {
	return &CommonTokenFactory{copyText: copyText}
}

// CommonTokenFactoryDEFAULT is the default CommonTokenFactory. It does not
// explicitly copy token text when constructing tokens.
var CommonTokenFactoryDEFAULT = NewCommonTokenFactory(false)

func (c *CommonTokenFactory) Create(source *TokenSourceCharStreamPair, ttype int, text string, channel, start, stop, line, column int) Token {
	t := NewCommonToken(source, ttype, channel, start, stop)

	t.line = line
	t.column = column

	if text != "" {
		t.SetText(text)
	} else if c.copyText && source.charStream != nil {
		t.SetText(source.charStream.GetTextFromInterval(NewInterval(start, stop)))
	}

	return t
}

func (c *CommonTokenFactory) createThin(ttype int, text string) Token {
	t := NewCommonToken(nil, ttype, TokenDefaultChannel, -1, -1)
	t.SetText(text)

	return t
}
