// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

type TokenStream interface {
	IntStream

	LT(k int) Token
	Reset()

	Get(index int) Token
	GetTokenSource() TokenSource
	SetTokenSource(TokenSource)

	GetAllText() string
	GetTextFromInterval(Interval) string
	GetTextFromRuleContext(RuleContext) string
	GetTextFromTokens(Token, Token) string
}
