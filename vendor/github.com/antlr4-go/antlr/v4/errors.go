// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

// The root of the ANTLR exception hierarchy. In general, ANTLR tracks just
//  3 kinds of errors: prediction errors, failed predicate errors, and
//  mismatched input errors. In each case, the parser knows where it is
//  in the input, where it is in the ATN, the rule invocation stack,
//  and what kind of problem occurred.

type RecognitionException interface {
	GetOffendingToken() Token
	GetMessage() string
	GetInputStream() IntStream
}

type BaseRecognitionException struct {
	message        string
	recognizer     Recognizer
	offendingToken Token
	offendingState int
	ctx            RuleContext
	input          IntStream
}

func NewBaseRecognitionException(message string, recognizer Recognizer, input IntStream, ctx RuleContext) *BaseRecognitionException {

	// todo
	//	Error.call(this)
	//
	//	if (!!Error.captureStackTrace) {
	//        Error.captureStackTrace(this, RecognitionException)
	//	} else {
	//		stack := NewError().stack
	//	}
	// TODO: may be able to use - "runtime" func Stack(buf []byte, all bool) int

	t := new(BaseRecognitionException)

	t.message = message
	t.recognizer = recognizer
	t.input = input
	t.ctx = ctx

	// The current Token when an error occurred. Since not all streams
	// support accessing symbols by index, we have to track the {@link Token}
	// instance itself.
	//
	t.offendingToken = nil

	// Get the ATN state number the parser was in at the time the error
	// occurred. For NoViableAltException and LexerNoViableAltException exceptions, this is the
	// DecisionState number. For others, it is the state whose outgoing edge we couldn't Match.
	//
	t.offendingState = -1
	if t.recognizer != nil {
		t.offendingState = t.recognizer.GetState()
	}

	return t
}

func (b *BaseRecognitionException) GetMessage() string {
	return b.message
}

func (b *BaseRecognitionException) GetOffendingToken() Token {
	return b.offendingToken
}

func (b *BaseRecognitionException) GetInputStream() IntStream {
	return b.input
}

// <p>If the state number is not known, b method returns -1.</p>

// getExpectedTokens gets the set of input symbols which could potentially follow the
// previously Matched symbol at the time this exception was raised.
//
// If the set of expected tokens is not known and could not be computed,
// this method returns nil.
//
// The func returns the set of token types that could potentially follow the current
// state in the {ATN}, or nil if the information is not available.

func (b *BaseRecognitionException) getExpectedTokens() *IntervalSet {
	if b.recognizer != nil {
		return b.recognizer.GetATN().getExpectedTokens(b.offendingState, b.ctx)
	}

	return nil
}

func (b *BaseRecognitionException) String() string {
	return b.message
}

type LexerNoViableAltException struct {
	*BaseRecognitionException

	startIndex     int
	deadEndConfigs *ATNConfigSet
}

func NewLexerNoViableAltException(lexer Lexer, input CharStream, startIndex int, deadEndConfigs *ATNConfigSet) *LexerNoViableAltException {

	l := new(LexerNoViableAltException)

	l.BaseRecognitionException = NewBaseRecognitionException("", lexer, input, nil)

	l.startIndex = startIndex
	l.deadEndConfigs = deadEndConfigs

	return l
}

func (l *LexerNoViableAltException) String() string {
	symbol := ""
	if l.startIndex >= 0 && l.startIndex < l.input.Size() {
		symbol = l.input.(CharStream).GetTextFromInterval(NewInterval(l.startIndex, l.startIndex))
	}
	return "LexerNoViableAltException" + symbol
}

type NoViableAltException struct {
	*BaseRecognitionException

	startToken     Token
	offendingToken Token
	ctx            ParserRuleContext
	deadEndConfigs *ATNConfigSet
}

// NewNoViableAltException creates an exception indicating that the parser could not decide which of two or more paths
// to take based upon the remaining input. It tracks the starting token
// of the offending input and also knows where the parser was
// in the various paths when the error.
//
// Reported by [ReportNoViableAlternative]
func NewNoViableAltException(recognizer Parser, input TokenStream, startToken Token, offendingToken Token, deadEndConfigs *ATNConfigSet, ctx ParserRuleContext) *NoViableAltException {

	if ctx == nil {
		ctx = recognizer.GetParserRuleContext()
	}

	if offendingToken == nil {
		offendingToken = recognizer.GetCurrentToken()
	}

	if startToken == nil {
		startToken = recognizer.GetCurrentToken()
	}

	if input == nil {
		input = recognizer.GetInputStream().(TokenStream)
	}

	n := new(NoViableAltException)
	n.BaseRecognitionException = NewBaseRecognitionException("", recognizer, input, ctx)

	// Which configurations did we try at input.Index() that couldn't Match
	// input.LT(1)
	n.deadEndConfigs = deadEndConfigs

	// The token object at the start index the input stream might
	// not be buffering tokens so get a reference to it.
	//
	// At the time the error occurred, of course the stream needs to keep a
	// buffer of all the tokens, but later we might not have access to those.
	n.startToken = startToken
	n.offendingToken = offendingToken

	return n
}

type InputMisMatchException struct {
	*BaseRecognitionException
}

// NewInputMisMatchException creates an exception that signifies any kind of mismatched input exceptions such as
// when the current input does not Match the expected token.
func NewInputMisMatchException(recognizer Parser) *InputMisMatchException {

	i := new(InputMisMatchException)
	i.BaseRecognitionException = NewBaseRecognitionException("", recognizer, recognizer.GetInputStream(), recognizer.GetParserRuleContext())

	i.offendingToken = recognizer.GetCurrentToken()

	return i

}

// FailedPredicateException indicates that a semantic predicate failed during validation. Validation of predicates
// occurs when normally parsing the alternative just like Matching a token.
// Disambiguating predicate evaluation occurs when we test a predicate during
// prediction.
type FailedPredicateException struct {
	*BaseRecognitionException

	ruleIndex      int
	predicateIndex int
	predicate      string
}

//goland:noinspection GoUnusedExportedFunction
func NewFailedPredicateException(recognizer Parser, predicate string, message string) *FailedPredicateException {

	f := new(FailedPredicateException)

	f.BaseRecognitionException = NewBaseRecognitionException(f.formatMessage(predicate, message), recognizer, recognizer.GetInputStream(), recognizer.GetParserRuleContext())

	s := recognizer.GetInterpreter().atn.states[recognizer.GetState()]
	trans := s.GetTransitions()[0]
	if trans2, ok := trans.(*PredicateTransition); ok {
		f.ruleIndex = trans2.ruleIndex
		f.predicateIndex = trans2.predIndex
	} else {
		f.ruleIndex = 0
		f.predicateIndex = 0
	}
	f.predicate = predicate
	f.offendingToken = recognizer.GetCurrentToken()

	return f
}

func (f *FailedPredicateException) formatMessage(predicate, message string) string {
	if message != "" {
		return message
	}

	return "failed predicate: {" + predicate + "}?"
}

type ParseCancellationException struct {
}

func (p ParseCancellationException) GetOffendingToken() Token {
	//TODO implement me
	panic("implement me")
}

func (p ParseCancellationException) GetMessage() string {
	//TODO implement me
	panic("implement me")
}

func (p ParseCancellationException) GetInputStream() IntStream {
	//TODO implement me
	panic("implement me")
}

func NewParseCancellationException() *ParseCancellationException {
	//	Error.call(this)
	//	Error.captureStackTrace(this, ParseCancellationException)
	return new(ParseCancellationException)
}
