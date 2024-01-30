// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
	"os"
	"strconv"
)

// Provides an empty default implementation of {@link ANTLRErrorListener}. The
// default implementation of each method does nothing, but can be overridden as
// necessary.

type ErrorListener interface {
	SyntaxError(recognizer Recognizer, offendingSymbol interface{}, line, column int, msg string, e RecognitionException)
	ReportAmbiguity(recognizer Parser, dfa *DFA, startIndex, stopIndex int, exact bool, ambigAlts *BitSet, configs ATNConfigSet)
	ReportAttemptingFullContext(recognizer Parser, dfa *DFA, startIndex, stopIndex int, conflictingAlts *BitSet, configs ATNConfigSet)
	ReportContextSensitivity(recognizer Parser, dfa *DFA, startIndex, stopIndex, prediction int, configs ATNConfigSet)
}

type DefaultErrorListener struct {
}

func NewDefaultErrorListener() *DefaultErrorListener {
	return new(DefaultErrorListener)
}

func (d *DefaultErrorListener) SyntaxError(recognizer Recognizer, offendingSymbol interface{}, line, column int, msg string, e RecognitionException) {
}

func (d *DefaultErrorListener) ReportAmbiguity(recognizer Parser, dfa *DFA, startIndex, stopIndex int, exact bool, ambigAlts *BitSet, configs ATNConfigSet) {
}

func (d *DefaultErrorListener) ReportAttemptingFullContext(recognizer Parser, dfa *DFA, startIndex, stopIndex int, conflictingAlts *BitSet, configs ATNConfigSet) {
}

func (d *DefaultErrorListener) ReportContextSensitivity(recognizer Parser, dfa *DFA, startIndex, stopIndex, prediction int, configs ATNConfigSet) {
}

type ConsoleErrorListener struct {
	*DefaultErrorListener
}

func NewConsoleErrorListener() *ConsoleErrorListener {
	return new(ConsoleErrorListener)
}

// Provides a default instance of {@link ConsoleErrorListener}.
var ConsoleErrorListenerINSTANCE = NewConsoleErrorListener()

// {@inheritDoc}
//
// <p>
// This implementation prints messages to {@link System//err} containing the
// values of {@code line}, {@code charPositionInLine}, and {@code msg} using
// the following format.</p>
//
// <pre>
// line <em>line</em>:<em>charPositionInLine</em> <em>msg</em>
// </pre>
func (c *ConsoleErrorListener) SyntaxError(recognizer Recognizer, offendingSymbol interface{}, line, column int, msg string, e RecognitionException) {
	fmt.Fprintln(os.Stderr, "line "+strconv.Itoa(line)+":"+strconv.Itoa(column)+" "+msg)
}

type ProxyErrorListener struct {
	*DefaultErrorListener
	delegates []ErrorListener
}

func NewProxyErrorListener(delegates []ErrorListener) *ProxyErrorListener {
	if delegates == nil {
		panic("delegates is not provided")
	}
	l := new(ProxyErrorListener)
	l.delegates = delegates
	return l
}

func (p *ProxyErrorListener) SyntaxError(recognizer Recognizer, offendingSymbol interface{}, line, column int, msg string, e RecognitionException) {
	for _, d := range p.delegates {
		d.SyntaxError(recognizer, offendingSymbol, line, column, msg, e)
	}
}

func (p *ProxyErrorListener) ReportAmbiguity(recognizer Parser, dfa *DFA, startIndex, stopIndex int, exact bool, ambigAlts *BitSet, configs ATNConfigSet) {
	for _, d := range p.delegates {
		d.ReportAmbiguity(recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs)
	}
}

func (p *ProxyErrorListener) ReportAttemptingFullContext(recognizer Parser, dfa *DFA, startIndex, stopIndex int, conflictingAlts *BitSet, configs ATNConfigSet) {
	for _, d := range p.delegates {
		d.ReportAttemptingFullContext(recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs)
	}
}

func (p *ProxyErrorListener) ReportContextSensitivity(recognizer Parser, dfa *DFA, startIndex, stopIndex, prediction int, configs ATNConfigSet) {
	for _, d := range p.delegates {
		d.ReportContextSensitivity(recognizer, dfa, startIndex, stopIndex, prediction, configs)
	}
}
