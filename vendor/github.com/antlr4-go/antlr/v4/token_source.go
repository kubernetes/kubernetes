// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

type TokenSource interface {
	NextToken() Token
	Skip()
	More()
	GetLine() int
	GetCharPositionInLine() int
	GetInputStream() CharStream
	GetSourceName() string
	setTokenFactory(factory TokenFactory)
	GetTokenFactory() TokenFactory
}
