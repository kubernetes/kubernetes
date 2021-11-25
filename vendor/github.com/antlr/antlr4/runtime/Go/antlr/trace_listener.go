// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import "fmt"

type TraceListener struct {
	parser *BaseParser
}

func NewTraceListener(parser *BaseParser) *TraceListener {
	tl := new(TraceListener)
	tl.parser = parser
	return tl
}

func (t *TraceListener) VisitErrorNode(_ ErrorNode) {
}

func (t *TraceListener) EnterEveryRule(ctx ParserRuleContext) {
	fmt.Println("enter   " + t.parser.GetRuleNames()[ctx.GetRuleIndex()] + ", LT(1)=" + t.parser.input.LT(1).GetText())
}

func (t *TraceListener) VisitTerminal(node TerminalNode) {
	fmt.Println("consume " + fmt.Sprint(node.GetSymbol()) + " rule " + t.parser.GetRuleNames()[t.parser.ctx.GetRuleIndex()])
}

func (t *TraceListener) ExitEveryRule(ctx ParserRuleContext) {
	fmt.Println("exit    " + t.parser.GetRuleNames()[ctx.GetRuleIndex()] + ", LT(1)=" + t.parser.input.LT(1).GetText())
}
