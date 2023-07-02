// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

type CharStream interface {
	IntStream
	GetText(int, int) string
	GetTextFromTokens(start, end Token) string
	GetTextFromInterval(*Interval) string
}
