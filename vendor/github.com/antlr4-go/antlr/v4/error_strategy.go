// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

type ErrorStrategy interface {
	reset(Parser)
	RecoverInline(Parser) Token
	Recover(Parser, RecognitionException)
	Sync(Parser)
	InErrorRecoveryMode(Parser) bool
	ReportError(Parser, RecognitionException)
	ReportMatch(Parser)
}

// DefaultErrorStrategy is the default implementation of ANTLRErrorStrategy used for
// error reporting and recovery in ANTLR parsers.
type DefaultErrorStrategy struct {
	errorRecoveryMode bool
	lastErrorIndex    int
	lastErrorStates   *IntervalSet
}

var _ ErrorStrategy = &DefaultErrorStrategy{}

func NewDefaultErrorStrategy() *DefaultErrorStrategy {

	d := new(DefaultErrorStrategy)

	// Indicates whether the error strategy is currently "recovering from an
	// error". This is used to suppress Reporting multiple error messages while
	// attempting to recover from a detected syntax error.
	//
	// @see //InErrorRecoveryMode
	//
	d.errorRecoveryMode = false

	// The index into the input stream where the last error occurred.
	// This is used to prevent infinite loops where an error is found
	// but no token is consumed during recovery...another error is found,
	// ad nauseam. This is a failsafe mechanism to guarantee that at least
	// one token/tree node is consumed for two errors.
	//
	d.lastErrorIndex = -1
	d.lastErrorStates = nil
	return d
}

// <p>The default implementation simply calls {@link //endErrorCondition} to
// ensure that the handler is not in error recovery mode.</p>
func (d *DefaultErrorStrategy) reset(recognizer Parser) {
	d.endErrorCondition(recognizer)
}

// This method is called to enter error recovery mode when a recognition
// exception is Reported.
func (d *DefaultErrorStrategy) beginErrorCondition(_ Parser) {
	d.errorRecoveryMode = true
}

func (d *DefaultErrorStrategy) InErrorRecoveryMode(_ Parser) bool {
	return d.errorRecoveryMode
}

// This method is called to leave error recovery mode after recovering from
// a recognition exception.
func (d *DefaultErrorStrategy) endErrorCondition(_ Parser) {
	d.errorRecoveryMode = false
	d.lastErrorStates = nil
	d.lastErrorIndex = -1
}

// ReportMatch is the default implementation of error matching and simply calls endErrorCondition.
func (d *DefaultErrorStrategy) ReportMatch(recognizer Parser) {
	d.endErrorCondition(recognizer)
}

// ReportError is the default implementation of error reporting.
// It returns immediately if the handler is already
// in error recovery mode. Otherwise, it calls [beginErrorCondition]
// and dispatches the Reporting task based on the runtime type of e
// according to the following table.
//
//		 [NoViableAltException]     : Dispatches the call to [ReportNoViableAlternative]
//		 [InputMisMatchException]   : Dispatches the call to [ReportInputMisMatch]
//	  [FailedPredicateException] : Dispatches the call to [ReportFailedPredicate]
//	  All other types            : Calls [NotifyErrorListeners] to Report the exception
func (d *DefaultErrorStrategy) ReportError(recognizer Parser, e RecognitionException) {
	// if we've already Reported an error and have not Matched a token
	// yet successfully, don't Report any errors.
	if d.InErrorRecoveryMode(recognizer) {
		return // don't Report spurious errors
	}
	d.beginErrorCondition(recognizer)

	switch t := e.(type) {
	default:
		fmt.Println("unknown recognition error type: " + reflect.TypeOf(e).Name())
		//            fmt.Println(e.stack)
		recognizer.NotifyErrorListeners(e.GetMessage(), e.GetOffendingToken(), e)
	case *NoViableAltException:
		d.ReportNoViableAlternative(recognizer, t)
	case *InputMisMatchException:
		d.ReportInputMisMatch(recognizer, t)
	case *FailedPredicateException:
		d.ReportFailedPredicate(recognizer, t)
	}
}

// Recover is the default recovery implementation.
// It reSynchronizes the parser by consuming tokens until we find one in the reSynchronization set -
// loosely the set of tokens that can follow the current rule.
func (d *DefaultErrorStrategy) Recover(recognizer Parser, _ RecognitionException) {

	if d.lastErrorIndex == recognizer.GetInputStream().Index() &&
		d.lastErrorStates != nil && d.lastErrorStates.contains(recognizer.GetState()) {
		// uh oh, another error at same token index and previously-Visited
		// state in ATN must be a case where LT(1) is in the recovery
		// token set so nothing got consumed. Consume a single token
		// at least to prevent an infinite loop d is a failsafe.
		recognizer.Consume()
	}
	d.lastErrorIndex = recognizer.GetInputStream().Index()
	if d.lastErrorStates == nil {
		d.lastErrorStates = NewIntervalSet()
	}
	d.lastErrorStates.addOne(recognizer.GetState())
	followSet := d.GetErrorRecoverySet(recognizer)
	d.consumeUntil(recognizer, followSet)
}

// Sync is the default implementation of error strategy synchronization.
//
// This Sync makes sure that the current lookahead symbol is consistent with what were expecting
// at this point in the [ATN]. You can call this anytime but ANTLR only
// generates code to check before sub-rules/loops and each iteration.
//
// Implements [Jim Idle]'s magic Sync mechanism in closures and optional
// sub-rules. E.g.:
//
//	a    : Sync ( stuff Sync )*
//	Sync : {consume to what can follow Sync}
//
// At the start of a sub-rule upon error, Sync performs single
// token deletion, if possible. If it can't do that, it bails on the current
// rule and uses the default error recovery, which consumes until the
// reSynchronization set of the current rule.
//
// If the sub-rule is optional
//
//	({@code (...)?}, {@code (...)*},
//
// or a block with an empty alternative), then the expected set includes what follows
// the sub-rule.
//
// During loop iteration, it consumes until it sees a token that can start a
// sub-rule or what follows loop. Yes, that is pretty aggressive. We opt to
// stay in the loop as long as possible.
//
// # Origins
//
// Previous versions of ANTLR did a poor job of their recovery within loops.
// A single mismatch token or missing token would force the parser to bail
// out of the entire rules surrounding the loop. So, for rule:
//
//	classfunc : 'class' ID '{' member* '}'
//
// input with an extra token between members would force the parser to
// consume until it found the next class definition rather than the next
// member definition of the current class.
//
// This functionality cost a bit of effort because the parser has to
// compare the token set at the start of the loop and at each iteration. If for
// some reason speed is suffering for you, you can turn off this
// functionality by simply overriding this method as empty:
//
//	{ }
//
// [Jim Idle]: https://github.com/jimidle
func (d *DefaultErrorStrategy) Sync(recognizer Parser) {
	// If already recovering, don't try to Sync
	if d.InErrorRecoveryMode(recognizer) {
		return
	}

	s := recognizer.GetInterpreter().atn.states[recognizer.GetState()]
	la := recognizer.GetTokenStream().LA(1)

	// try cheaper subset first might get lucky. seems to shave a wee bit off
	nextTokens := recognizer.GetATN().NextTokens(s, nil)
	if nextTokens.contains(TokenEpsilon) || nextTokens.contains(la) {
		return
	}

	switch s.GetStateType() {
	case ATNStateBlockStart, ATNStateStarBlockStart, ATNStatePlusBlockStart, ATNStateStarLoopEntry:
		// Report error and recover if possible
		if d.SingleTokenDeletion(recognizer) != nil {
			return
		}
		recognizer.SetError(NewInputMisMatchException(recognizer))
	case ATNStatePlusLoopBack, ATNStateStarLoopBack:
		d.ReportUnwantedToken(recognizer)
		expecting := NewIntervalSet()
		expecting.addSet(recognizer.GetExpectedTokens())
		whatFollowsLoopIterationOrRule := expecting.addSet(d.GetErrorRecoverySet(recognizer))
		d.consumeUntil(recognizer, whatFollowsLoopIterationOrRule)
	default:
		// do nothing if we can't identify the exact kind of ATN state
	}
}

// ReportNoViableAlternative is called by [ReportError] when the exception is a [NoViableAltException].
//
// See also [ReportError]
func (d *DefaultErrorStrategy) ReportNoViableAlternative(recognizer Parser, e *NoViableAltException) {
	tokens := recognizer.GetTokenStream()
	var input string
	if tokens != nil {
		if e.startToken.GetTokenType() == TokenEOF {
			input = "<EOF>"
		} else {
			input = tokens.GetTextFromTokens(e.startToken, e.offendingToken)
		}
	} else {
		input = "<unknown input>"
	}
	msg := "no viable alternative at input " + d.escapeWSAndQuote(input)
	recognizer.NotifyErrorListeners(msg, e.offendingToken, e)
}

// ReportInputMisMatch is called by [ReportError] when the exception is an [InputMisMatchException]
//
// See also: [ReportError]
func (d *DefaultErrorStrategy) ReportInputMisMatch(recognizer Parser, e *InputMisMatchException) {
	msg := "mismatched input " + d.GetTokenErrorDisplay(e.offendingToken) +
		" expecting " + e.getExpectedTokens().StringVerbose(recognizer.GetLiteralNames(), recognizer.GetSymbolicNames(), false)
	recognizer.NotifyErrorListeners(msg, e.offendingToken, e)
}

// ReportFailedPredicate is called by [ReportError] when the exception is a [FailedPredicateException].
//
// See also: [ReportError]
func (d *DefaultErrorStrategy) ReportFailedPredicate(recognizer Parser, e *FailedPredicateException) {
	ruleName := recognizer.GetRuleNames()[recognizer.GetParserRuleContext().GetRuleIndex()]
	msg := "rule " + ruleName + " " + e.message
	recognizer.NotifyErrorListeners(msg, e.offendingToken, e)
}

// ReportUnwantedToken is called to report a syntax error that requires the removal
// of a token from the input stream. At the time d method is called, the
// erroneous symbol is the current LT(1) symbol and has not yet been
// removed from the input stream. When this method returns,
// recognizer is in error recovery mode.
//
// This method is called when singleTokenDeletion identifies
// single-token deletion as a viable recovery strategy for a mismatched
// input error.
//
// The default implementation simply returns if the handler is already in
// error recovery mode. Otherwise, it calls beginErrorCondition to
// enter error recovery mode, followed by calling
// [NotifyErrorListeners]
func (d *DefaultErrorStrategy) ReportUnwantedToken(recognizer Parser) {
	if d.InErrorRecoveryMode(recognizer) {
		return
	}
	d.beginErrorCondition(recognizer)
	t := recognizer.GetCurrentToken()
	tokenName := d.GetTokenErrorDisplay(t)
	expecting := d.GetExpectedTokens(recognizer)
	msg := "extraneous input " + tokenName + " expecting " +
		expecting.StringVerbose(recognizer.GetLiteralNames(), recognizer.GetSymbolicNames(), false)
	recognizer.NotifyErrorListeners(msg, t, nil)
}

// ReportMissingToken is called to report a syntax error which requires the
// insertion of a missing token into the input stream. At the time this
// method is called, the missing token has not yet been inserted. When this
// method returns, recognizer is in error recovery mode.
//
// This method is called when singleTokenInsertion identifies
// single-token insertion as a viable recovery strategy for a mismatched
// input error.
//
// The default implementation simply returns if the handler is already in
// error recovery mode. Otherwise, it calls beginErrorCondition to
// enter error recovery mode, followed by calling [NotifyErrorListeners]
func (d *DefaultErrorStrategy) ReportMissingToken(recognizer Parser) {
	if d.InErrorRecoveryMode(recognizer) {
		return
	}
	d.beginErrorCondition(recognizer)
	t := recognizer.GetCurrentToken()
	expecting := d.GetExpectedTokens(recognizer)
	msg := "missing " + expecting.StringVerbose(recognizer.GetLiteralNames(), recognizer.GetSymbolicNames(), false) +
		" at " + d.GetTokenErrorDisplay(t)
	recognizer.NotifyErrorListeners(msg, t, nil)
}

// The RecoverInline default implementation attempts to recover from the mismatched input
// by using single token insertion and deletion as described below. If the
// recovery attempt fails, this method panics with [InputMisMatchException}.
// TODO: Not sure that panic() is the right thing to do here - JI
//
// # EXTRA TOKEN (single token deletion)
//
// LA(1) is not what we are looking for. If LA(2) has the
// right token, however, then assume LA(1) is some extra spurious
// token and delete it. Then consume and return the next token (which was
// the LA(2) token) as the successful result of the Match operation.
//
// # This recovery strategy is implemented by singleTokenDeletion
//
// # MISSING TOKEN (single token insertion)
//
// If current token  -at LA(1) - is consistent with what could come
// after the expected LA(1) token, then assume the token is missing
// and use the parser's [TokenFactory] to create it on the fly. The
// “insertion” is performed by returning the created token as the successful
// result of the Match operation.
//
// This recovery strategy is implemented by [SingleTokenInsertion].
//
// # Example
//
// For example, Input i=(3 is clearly missing the ')'. When
// the parser returns from the nested call to expr, it will have
// call the chain:
//
//	stat → expr → atom
//
// and it will be trying to Match the ')' at this point in the
// derivation:
//
//	  : ID '=' '(' INT ')' ('+' atom)* ';'
//		                ^
//
// The attempt to [Match] ')' will fail when it sees ';' and
// call [RecoverInline]. To recover, it sees that LA(1)==';'
// is in the set of tokens that can follow the ')' token reference
// in rule atom. It can assume that you forgot the ')'.
func (d *DefaultErrorStrategy) RecoverInline(recognizer Parser) Token {
	// SINGLE TOKEN DELETION
	MatchedSymbol := d.SingleTokenDeletion(recognizer)
	if MatchedSymbol != nil {
		// we have deleted the extra token.
		// now, move past ttype token as if all were ok
		recognizer.Consume()
		return MatchedSymbol
	}
	// SINGLE TOKEN INSERTION
	if d.SingleTokenInsertion(recognizer) {
		return d.GetMissingSymbol(recognizer)
	}
	// even that didn't work must panic the exception
	recognizer.SetError(NewInputMisMatchException(recognizer))
	return nil
}

// SingleTokenInsertion implements the single-token insertion inline error recovery
// strategy. It is called by [RecoverInline] if the single-token
// deletion strategy fails to recover from the mismatched input. If this
// method returns {@code true}, {@code recognizer} will be in error recovery
// mode.
//
// This method determines whether single-token insertion is viable by
// checking if the LA(1) input symbol could be successfully Matched
// if it were instead the LA(2) symbol. If this method returns
// {@code true}, the caller is responsible for creating and inserting a
// token with the correct type to produce this behavior.</p>
//
// This func returns true if single-token insertion is a viable recovery
// strategy for the current mismatched input.
func (d *DefaultErrorStrategy) SingleTokenInsertion(recognizer Parser) bool {
	currentSymbolType := recognizer.GetTokenStream().LA(1)
	// if current token is consistent with what could come after current
	// ATN state, then we know we're missing a token error recovery
	// is free to conjure up and insert the missing token
	atn := recognizer.GetInterpreter().atn
	currentState := atn.states[recognizer.GetState()]
	next := currentState.GetTransitions()[0].getTarget()
	expectingAtLL2 := atn.NextTokens(next, recognizer.GetParserRuleContext())
	if expectingAtLL2.contains(currentSymbolType) {
		d.ReportMissingToken(recognizer)
		return true
	}

	return false
}

// SingleTokenDeletion implements the single-token deletion inline error recovery
// strategy. It is called by [RecoverInline] to attempt to recover
// from mismatched input. If this method returns nil, the parser and error
// handler state will not have changed. If this method returns non-nil,
// recognizer will not be in error recovery mode since the
// returned token was a successful Match.
//
// If the single-token deletion is successful, this method calls
// [ReportUnwantedToken] to Report the error, followed by
// [Consume] to actually “delete” the extraneous token. Then,
// before returning, [ReportMatch] is called to signal a successful
// Match.
//
// The func returns the successfully Matched [Token] instance if single-token
// deletion successfully recovers from the mismatched input, otherwise nil.
func (d *DefaultErrorStrategy) SingleTokenDeletion(recognizer Parser) Token {
	NextTokenType := recognizer.GetTokenStream().LA(2)
	expecting := d.GetExpectedTokens(recognizer)
	if expecting.contains(NextTokenType) {
		d.ReportUnwantedToken(recognizer)
		// print("recoverFromMisMatchedToken deleting " \
		// + str(recognizer.GetTokenStream().LT(1)) \
		// + " since " + str(recognizer.GetTokenStream().LT(2)) \
		// + " is what we want", file=sys.stderr)
		recognizer.Consume() // simply delete extra token
		// we want to return the token we're actually Matching
		MatchedSymbol := recognizer.GetCurrentToken()
		d.ReportMatch(recognizer) // we know current token is correct
		return MatchedSymbol
	}

	return nil
}

// GetMissingSymbol conjures up a missing token during error recovery.
//
// The recognizer attempts to recover from single missing
// symbols. But, actions might refer to that missing symbol.
// For example:
//
//	x=ID {f($x)}.
//
// The action clearly assumes
// that there has been an identifier Matched previously and that
// $x points at that token. If that token is missing, but
// the next token in the stream is what we want we assume that
// this token is missing, and we keep going. Because we
// have to return some token to replace the missing token,
// we have to conjure one up. This method gives the user control
// over the tokens returned for missing tokens. Mostly,
// you will want to create something special for identifier
// tokens. For literals such as '{' and ',', the default
// action in the parser or tree parser works. It simply creates
// a [CommonToken] of the appropriate type. The text will be the token name.
// If you need to change which tokens must be created by the lexer,
// override this method to create the appropriate tokens.
func (d *DefaultErrorStrategy) GetMissingSymbol(recognizer Parser) Token {
	currentSymbol := recognizer.GetCurrentToken()
	expecting := d.GetExpectedTokens(recognizer)
	expectedTokenType := expecting.first()
	var tokenText string

	if expectedTokenType == TokenEOF {
		tokenText = "<missing EOF>"
	} else {
		ln := recognizer.GetLiteralNames()
		if expectedTokenType > 0 && expectedTokenType < len(ln) {
			tokenText = "<missing " + recognizer.GetLiteralNames()[expectedTokenType] + ">"
		} else {
			tokenText = "<missing undefined>" // TODO: matches the JS impl
		}
	}
	current := currentSymbol
	lookback := recognizer.GetTokenStream().LT(-1)
	if current.GetTokenType() == TokenEOF && lookback != nil {
		current = lookback
	}

	tf := recognizer.GetTokenFactory()

	return tf.Create(current.GetSource(), expectedTokenType, tokenText, TokenDefaultChannel, -1, -1, current.GetLine(), current.GetColumn())
}

func (d *DefaultErrorStrategy) GetExpectedTokens(recognizer Parser) *IntervalSet {
	return recognizer.GetExpectedTokens()
}

// GetTokenErrorDisplay determines how  a token should be displayed in an error message.
// The default is to display just the text, but during development you might
// want to have a lot of information spit out. Override this func in that case
// to use t.String() (which, for [CommonToken], dumps everything about
// the token). This is better than forcing you to override a method in
// your token objects because you don't have to go modify your lexer
// so that it creates a new type.
func (d *DefaultErrorStrategy) GetTokenErrorDisplay(t Token) string {
	if t == nil {
		return "<no token>"
	}
	s := t.GetText()
	if s == "" {
		if t.GetTokenType() == TokenEOF {
			s = "<EOF>"
		} else {
			s = "<" + strconv.Itoa(t.GetTokenType()) + ">"
		}
	}
	return d.escapeWSAndQuote(s)
}

func (d *DefaultErrorStrategy) escapeWSAndQuote(s string) string {
	s = strings.Replace(s, "\t", "\\t", -1)
	s = strings.Replace(s, "\n", "\\n", -1)
	s = strings.Replace(s, "\r", "\\r", -1)
	return "'" + s + "'"
}

// GetErrorRecoverySet computes the error recovery set for the current rule. During
// rule invocation, the parser pushes the set of tokens that can
// follow that rule reference on the stack. This amounts to
// computing FIRST of what follows the rule reference in the
// enclosing rule. See LinearApproximator.FIRST().
//
// This local follow set only includes tokens
// from within the rule i.e., the FIRST computation done by
// ANTLR stops at the end of a rule.
//
// # Example
//
// When you find a "no viable alt exception", the input is not
// consistent with any of the alternatives for rule r. The best
// thing to do is to consume tokens until you see something that
// can legally follow a call to r or any rule that called r.
// You don't want the exact set of viable next tokens because the
// input might just be missing a token--you might consume the
// rest of the input looking for one of the missing tokens.
//
// Consider the grammar:
//
//		a : '[' b ']'
//		  | '(' b ')'
//		  ;
//
//		b : c '^' INT
//	      ;
//
//		c : ID
//		  | INT
//		  ;
//
// At each rule invocation, the set of tokens that could follow
// that rule is pushed on a stack. Here are the various
// context-sensitive follow sets:
//
//	FOLLOW(b1_in_a) = FIRST(']') = ']'
//	FOLLOW(b2_in_a) = FIRST(')') = ')'
//	FOLLOW(c_in_b)  = FIRST('^') = '^'
//
// Upon erroneous input “[]”, the call chain is
//
//	a → b → c
//
// and, hence, the follow context stack is:
//
//	Depth Follow set   Start of rule execution
//	  0   <EOF>        a (from main())
//	  1   ']'          b
//	  2   '^'          c
//
// Notice that ')' is not included, because b would have to have
// been called from a different context in rule a for ')' to be
// included.
//
// For error recovery, we cannot consider FOLLOW(c)
// (context-sensitive or otherwise). We need the combined set of
// all context-sensitive FOLLOW sets - the set of all tokens that
// could follow any reference in the call chain. We need to
// reSync to one of those tokens. Note that FOLLOW(c)='^' and if
// we reSync'd to that token, we'd consume until EOF. We need to
// Sync to context-sensitive FOLLOWs for a, b, and c:
//
//	{']','^'}
//
// In this case, for input "[]", LA(1) is ']' and in the set, so we would
// not consume anything. After printing an error, rule c would
// return normally. Rule b would not find the required '^' though.
// At this point, it gets a mismatched token error and panics an
// exception (since LA(1) is not in the viable following token
// set). The rule exception handler tries to recover, but finds
// the same recovery set and doesn't consume anything. Rule b
// exits normally returning to rule a. Now it finds the ']' (and
// with the successful Match exits errorRecovery mode).
//
// So, you can see that the parser walks up the call chain looking
// for the token that was a member of the recovery set.
//
// Errors are not generated in errorRecovery mode.
//
// ANTLR's error recovery mechanism is based upon original ideas:
//
// [Algorithms + Data Structures = Programs] by Niklaus Wirth and
// [A note on error recovery in recursive descent parsers].
//
// Later, Josef Grosch had some good ideas in [Efficient and Comfortable Error Recovery in Recursive Descent
// Parsers]
//
// Like Grosch I implement context-sensitive FOLLOW sets that are combined  at run-time upon error to avoid overhead
// during parsing. Later, the runtime Sync was improved for loops/sub-rules see [Sync] docs
//
// [A note on error recovery in recursive descent parsers]: http://portal.acm.org/citation.cfm?id=947902.947905
// [Algorithms + Data Structures = Programs]: https://t.ly/5QzgE
// [Efficient and Comfortable Error Recovery in Recursive Descent Parsers]: ftp://www.cocolab.com/products/cocktail/doca4.ps/ell.ps.zip
func (d *DefaultErrorStrategy) GetErrorRecoverySet(recognizer Parser) *IntervalSet {
	atn := recognizer.GetInterpreter().atn
	ctx := recognizer.GetParserRuleContext()
	recoverSet := NewIntervalSet()
	for ctx != nil && ctx.GetInvokingState() >= 0 {
		// compute what follows who invoked us
		invokingState := atn.states[ctx.GetInvokingState()]
		rt := invokingState.GetTransitions()[0]
		follow := atn.NextTokens(rt.(*RuleTransition).followState, nil)
		recoverSet.addSet(follow)
		ctx = ctx.GetParent().(ParserRuleContext)
	}
	recoverSet.removeOne(TokenEpsilon)
	return recoverSet
}

// Consume tokens until one Matches the given token set.//
func (d *DefaultErrorStrategy) consumeUntil(recognizer Parser, set *IntervalSet) {
	ttype := recognizer.GetTokenStream().LA(1)
	for ttype != TokenEOF && !set.contains(ttype) {
		recognizer.Consume()
		ttype = recognizer.GetTokenStream().LA(1)
	}
}

// The BailErrorStrategy implementation of ANTLRErrorStrategy responds to syntax errors
// by immediately canceling the parse operation with a
// [ParseCancellationException]. The implementation ensures that the
// [ParserRuleContext//exception] field is set for all parse tree nodes
// that were not completed prior to encountering the error.
//
// This error strategy is useful in the following scenarios.
//
//   - Two-stage parsing: This error strategy allows the first
//     stage of two-stage parsing to immediately terminate if an error is
//     encountered, and immediately fall back to the second stage. In addition to
//     avoiding wasted work by attempting to recover from errors here, the empty
//     implementation of [BailErrorStrategy.Sync] improves the performance of
//     the first stage.
//
//   - Silent validation: When syntax errors are not being
//     Reported or logged, and the parse result is simply ignored if errors occur,
//     the [BailErrorStrategy] avoids wasting work on recovering from errors
//     when the result will be ignored either way.
//
//     myparser.SetErrorHandler(NewBailErrorStrategy())
//
// See also: [Parser.SetErrorHandler(ANTLRErrorStrategy)]
type BailErrorStrategy struct {
	*DefaultErrorStrategy
}

var _ ErrorStrategy = &BailErrorStrategy{}

//goland:noinspection GoUnusedExportedFunction
func NewBailErrorStrategy() *BailErrorStrategy {

	b := new(BailErrorStrategy)

	b.DefaultErrorStrategy = NewDefaultErrorStrategy()

	return b
}

// Recover Instead of recovering from exception e, re-panic it wrapped
// in a [ParseCancellationException] so it is not caught by the
// rule func catches. Use Exception.GetCause() to get the
// original [RecognitionException].
func (b *BailErrorStrategy) Recover(recognizer Parser, e RecognitionException) {
	context := recognizer.GetParserRuleContext()
	for context != nil {
		context.SetException(e)
		if parent, ok := context.GetParent().(ParserRuleContext); ok {
			context = parent
		} else {
			context = nil
		}
	}
	recognizer.SetError(NewParseCancellationException()) // TODO: we don't emit e properly
}

// RecoverInline makes sure we don't attempt to recover inline if the parser
// successfully recovers, it won't panic an exception.
func (b *BailErrorStrategy) RecoverInline(recognizer Parser) Token {
	b.Recover(recognizer, NewInputMisMatchException(recognizer))

	return nil
}

// Sync makes sure we don't attempt to recover from problems in sub-rules.
func (b *BailErrorStrategy) Sync(_ Parser) {
}
