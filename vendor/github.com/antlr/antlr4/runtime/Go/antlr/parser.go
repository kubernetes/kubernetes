// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
	"strconv"
)

type Parser interface {
	Recognizer

	GetInterpreter() *ParserATNSimulator

	GetTokenStream() TokenStream
	GetTokenFactory() TokenFactory
	GetParserRuleContext() ParserRuleContext
	SetParserRuleContext(ParserRuleContext)
	Consume() Token
	GetParseListeners() []ParseTreeListener

	GetErrorHandler() ErrorStrategy
	SetErrorHandler(ErrorStrategy)
	GetInputStream() IntStream
	GetCurrentToken() Token
	GetExpectedTokens() *IntervalSet
	NotifyErrorListeners(string, Token, RecognitionException)
	IsExpectedToken(int) bool
	GetPrecedence() int
	GetRuleInvocationStack(ParserRuleContext) []string
}

type BaseParser struct {
	*BaseRecognizer

	Interpreter     *ParserATNSimulator
	BuildParseTrees bool

	input           TokenStream
	errHandler      ErrorStrategy
	precedenceStack IntStack
	ctx             ParserRuleContext

	tracer         *TraceListener
	parseListeners []ParseTreeListener
	_SyntaxErrors  int
}

// p.is all the parsing support code essentially most of it is error
// recovery stuff.//
func NewBaseParser(input TokenStream) *BaseParser {

	p := new(BaseParser)

	p.BaseRecognizer = NewBaseRecognizer()

	// The input stream.
	p.input = nil
	// The error handling strategy for the parser. The default value is a new
	// instance of {@link DefaultErrorStrategy}.
	p.errHandler = NewDefaultErrorStrategy()
	p.precedenceStack = make([]int, 0)
	p.precedenceStack.Push(0)
	// The {@link ParserRuleContext} object for the currently executing rule.
	// p.is always non-nil during the parsing process.
	p.ctx = nil
	// Specifies whether or not the parser should construct a parse tree during
	// the parsing process. The default value is {@code true}.
	p.BuildParseTrees = true
	// When {@link //setTrace}{@code (true)} is called, a reference to the
	// {@link TraceListener} is stored here so it can be easily removed in a
	// later call to {@link //setTrace}{@code (false)}. The listener itself is
	// implemented as a parser listener so p.field is not directly used by
	// other parser methods.
	p.tracer = nil
	// The list of {@link ParseTreeListener} listeners registered to receive
	// events during the parse.
	p.parseListeners = nil
	// The number of syntax errors Reported during parsing. p.value is
	// incremented each time {@link //NotifyErrorListeners} is called.
	p._SyntaxErrors = 0
	p.SetInputStream(input)

	return p
}

// p.field maps from the serialized ATN string to the deserialized {@link
// ATN} with
// bypass alternatives.
//
// @see ATNDeserializationOptions//isGenerateRuleBypassTransitions()
//
var bypassAltsAtnCache = make(map[string]int)

// reset the parser's state//
func (p *BaseParser) reset() {
	if p.input != nil {
		p.input.Seek(0)
	}
	p.errHandler.reset(p)
	p.ctx = nil
	p._SyntaxErrors = 0
	p.SetTrace(nil)
	p.precedenceStack = make([]int, 0)
	p.precedenceStack.Push(0)
	if p.Interpreter != nil {
		p.Interpreter.reset()
	}
}

func (p *BaseParser) GetErrorHandler() ErrorStrategy {
	return p.errHandler
}

func (p *BaseParser) SetErrorHandler(e ErrorStrategy) {
	p.errHandler = e
}

// Match current input symbol against {@code ttype}. If the symbol type
// Matches, {@link ANTLRErrorStrategy//ReportMatch} and {@link //consume} are
// called to complete the Match process.
//
// <p>If the symbol type does not Match,
// {@link ANTLRErrorStrategy//recoverInline} is called on the current error
// strategy to attempt recovery. If {@link //getBuildParseTree} is
// {@code true} and the token index of the symbol returned by
// {@link ANTLRErrorStrategy//recoverInline} is -1, the symbol is added to
// the parse tree by calling {@link ParserRuleContext//addErrorNode}.</p>
//
// @param ttype the token type to Match
// @return the Matched symbol
// @panics RecognitionException if the current input symbol did not Match
// {@code ttype} and the error strategy could not recover from the
// mismatched symbol

func (p *BaseParser) Match(ttype int) Token {

	t := p.GetCurrentToken()

	if t.GetTokenType() == ttype {
		p.errHandler.ReportMatch(p)
		p.Consume()
	} else {
		t = p.errHandler.RecoverInline(p)
		if p.BuildParseTrees && t.GetTokenIndex() == -1 {
			// we must have conjured up a Newtoken during single token
			// insertion
			// if it's not the current symbol
			p.ctx.AddErrorNode(t)
		}
	}

	return t
}

// Match current input symbol as a wildcard. If the symbol type Matches
// (i.e. has a value greater than 0), {@link ANTLRErrorStrategy//ReportMatch}
// and {@link //consume} are called to complete the Match process.
//
// <p>If the symbol type does not Match,
// {@link ANTLRErrorStrategy//recoverInline} is called on the current error
// strategy to attempt recovery. If {@link //getBuildParseTree} is
// {@code true} and the token index of the symbol returned by
// {@link ANTLRErrorStrategy//recoverInline} is -1, the symbol is added to
// the parse tree by calling {@link ParserRuleContext//addErrorNode}.</p>
//
// @return the Matched symbol
// @panics RecognitionException if the current input symbol did not Match
// a wildcard and the error strategy could not recover from the mismatched
// symbol

func (p *BaseParser) MatchWildcard() Token {
	t := p.GetCurrentToken()
	if t.GetTokenType() > 0 {
		p.errHandler.ReportMatch(p)
		p.Consume()
	} else {
		t = p.errHandler.RecoverInline(p)
		if p.BuildParseTrees && t.GetTokenIndex() == -1 {
			// we must have conjured up a Newtoken during single token
			// insertion
			// if it's not the current symbol
			p.ctx.AddErrorNode(t)
		}
	}
	return t
}

func (p *BaseParser) GetParserRuleContext() ParserRuleContext {
	return p.ctx
}

func (p *BaseParser) SetParserRuleContext(v ParserRuleContext) {
	p.ctx = v
}

func (p *BaseParser) GetParseListeners() []ParseTreeListener {
	if p.parseListeners == nil {
		return make([]ParseTreeListener, 0)
	}
	return p.parseListeners
}

// Registers {@code listener} to receive events during the parsing process.
//
// <p>To support output-preserving grammar transformations (including but not
// limited to left-recursion removal, automated left-factoring, and
// optimized code generation), calls to listener methods during the parse
// may differ substantially from calls made by
// {@link ParseTreeWalker//DEFAULT} used after the parse is complete. In
// particular, rule entry and exit events may occur in a different order
// during the parse than after the parser. In addition, calls to certain
// rule entry methods may be omitted.</p>
//
// <p>With the following specific exceptions, calls to listener events are
// <em>deterministic</em>, i.e. for identical input the calls to listener
// methods will be the same.</p>
//
// <ul>
// <li>Alterations to the grammar used to generate code may change the
// behavior of the listener calls.</li>
// <li>Alterations to the command line options passed to ANTLR 4 when
// generating the parser may change the behavior of the listener calls.</li>
// <li>Changing the version of the ANTLR Tool used to generate the parser
// may change the behavior of the listener calls.</li>
// </ul>
//
// @param listener the listener to add
//
// @panics nilPointerException if {@code} listener is {@code nil}
//
func (p *BaseParser) AddParseListener(listener ParseTreeListener) {
	if listener == nil {
		panic("listener")
	}
	if p.parseListeners == nil {
		p.parseListeners = make([]ParseTreeListener, 0)
	}
	p.parseListeners = append(p.parseListeners, listener)
}

//
// Remove {@code listener} from the list of parse listeners.
//
// <p>If {@code listener} is {@code nil} or has not been added as a parse
// listener, p.method does nothing.</p>
// @param listener the listener to remove
//
func (p *BaseParser) RemoveParseListener(listener ParseTreeListener) {

	if p.parseListeners != nil {

		idx := -1
		for i, v := range p.parseListeners {
			if v == listener {
				idx = i
				break
			}
		}

		if idx == -1 {
			return
		}

		// remove the listener from the slice
		p.parseListeners = append(p.parseListeners[0:idx], p.parseListeners[idx+1:]...)

		if len(p.parseListeners) == 0 {
			p.parseListeners = nil
		}
	}
}

// Remove all parse listeners.
func (p *BaseParser) removeParseListeners() {
	p.parseListeners = nil
}

// Notify any parse listeners of an enter rule event.
func (p *BaseParser) TriggerEnterRuleEvent() {
	if p.parseListeners != nil {
		ctx := p.ctx
		for _, listener := range p.parseListeners {
			listener.EnterEveryRule(ctx)
			ctx.EnterRule(listener)
		}
	}
}

//
// Notify any parse listeners of an exit rule event.
//
// @see //addParseListener
//
func (p *BaseParser) TriggerExitRuleEvent() {
	if p.parseListeners != nil {
		// reverse order walk of listeners
		ctx := p.ctx
		l := len(p.parseListeners) - 1

		for i := range p.parseListeners {
			listener := p.parseListeners[l-i]
			ctx.ExitRule(listener)
			listener.ExitEveryRule(ctx)
		}
	}
}

func (p *BaseParser) GetInterpreter() *ParserATNSimulator {
	return p.Interpreter
}

func (p *BaseParser) GetATN() *ATN {
	return p.Interpreter.atn
}

func (p *BaseParser) GetTokenFactory() TokenFactory {
	return p.input.GetTokenSource().GetTokenFactory()
}

// Tell our token source and error strategy about a Newway to create tokens.//
func (p *BaseParser) setTokenFactory(factory TokenFactory) {
	p.input.GetTokenSource().setTokenFactory(factory)
}

// The ATN with bypass alternatives is expensive to create so we create it
// lazily.
//
// @panics UnsupportedOperationException if the current parser does not
// implement the {@link //getSerializedATN()} method.
//
func (p *BaseParser) GetATNWithBypassAlts() {

	// TODO
	panic("Not implemented!")

	//	serializedAtn := p.getSerializedATN()
	//	if (serializedAtn == nil) {
	//		panic("The current parser does not support an ATN with bypass alternatives.")
	//	}
	//	result := p.bypassAltsAtnCache[serializedAtn]
	//	if (result == nil) {
	//		deserializationOptions := NewATNDeserializationOptions(nil)
	//		deserializationOptions.generateRuleBypassTransitions = true
	//		result = NewATNDeserializer(deserializationOptions).deserialize(serializedAtn)
	//		p.bypassAltsAtnCache[serializedAtn] = result
	//	}
	//	return result
}

// The preferred method of getting a tree pattern. For example, here's a
// sample use:
//
// <pre>
// ParseTree t = parser.expr()
// ParseTreePattern p = parser.compileParseTreePattern("&ltID&gt+0",
// MyParser.RULE_expr)
// ParseTreeMatch m = p.Match(t)
// String id = m.Get("ID")
// </pre>

func (p *BaseParser) compileParseTreePattern(pattern, patternRuleIndex, lexer Lexer) {

	panic("NewParseTreePatternMatcher not implemented!")
	//
	//	if (lexer == nil) {
	//		if (p.GetTokenStream() != nil) {
	//			tokenSource := p.GetTokenStream().GetTokenSource()
	//			if _, ok := tokenSource.(ILexer); ok {
	//				lexer = tokenSource
	//			}
	//		}
	//	}
	//	if (lexer == nil) {
	//		panic("Parser can't discover a lexer to use")
	//	}

	//	m := NewParseTreePatternMatcher(lexer, p)
	//	return m.compile(pattern, patternRuleIndex)
}

func (p *BaseParser) GetInputStream() IntStream {
	return p.GetTokenStream()
}

func (p *BaseParser) SetInputStream(input TokenStream) {
	p.SetTokenStream(input)
}

func (p *BaseParser) GetTokenStream() TokenStream {
	return p.input
}

// Set the token stream and reset the parser.//
func (p *BaseParser) SetTokenStream(input TokenStream) {
	p.input = nil
	p.reset()
	p.input = input
}

// Match needs to return the current input symbol, which gets put
// into the label for the associated token ref e.g., x=ID.
//
func (p *BaseParser) GetCurrentToken() Token {
	return p.input.LT(1)
}

func (p *BaseParser) NotifyErrorListeners(msg string, offendingToken Token, err RecognitionException) {
	if offendingToken == nil {
		offendingToken = p.GetCurrentToken()
	}
	p._SyntaxErrors++
	line := offendingToken.GetLine()
	column := offendingToken.GetColumn()
	listener := p.GetErrorListenerDispatch()
	listener.SyntaxError(p, offendingToken, line, column, msg, err)
}

func (p *BaseParser) Consume() Token {
	o := p.GetCurrentToken()
	if o.GetTokenType() != TokenEOF {
		p.GetInputStream().Consume()
	}
	hasListener := p.parseListeners != nil && len(p.parseListeners) > 0
	if p.BuildParseTrees || hasListener {
		if p.errHandler.InErrorRecoveryMode(p) {
			node := p.ctx.AddErrorNode(o)
			if p.parseListeners != nil {
				for _, l := range p.parseListeners {
					l.VisitErrorNode(node)
				}
			}

		} else {
			node := p.ctx.AddTokenNode(o)
			if p.parseListeners != nil {
				for _, l := range p.parseListeners {
					l.VisitTerminal(node)
				}
			}
		}
		//        node.invokingState = p.state
	}

	return o
}

func (p *BaseParser) addContextToParseTree() {
	// add current context to parent if we have a parent
	if p.ctx.GetParent() != nil {
		p.ctx.GetParent().(ParserRuleContext).AddChild(p.ctx)
	}
}

func (p *BaseParser) EnterRule(localctx ParserRuleContext, state, ruleIndex int) {
	p.SetState(state)
	p.ctx = localctx
	p.ctx.SetStart(p.input.LT(1))
	if p.BuildParseTrees {
		p.addContextToParseTree()
	}
	if p.parseListeners != nil {
		p.TriggerEnterRuleEvent()
	}
}

func (p *BaseParser) ExitRule() {
	p.ctx.SetStop(p.input.LT(-1))
	// trigger event on ctx, before it reverts to parent
	if p.parseListeners != nil {
		p.TriggerExitRuleEvent()
	}
	p.SetState(p.ctx.GetInvokingState())
	if p.ctx.GetParent() != nil {
		p.ctx = p.ctx.GetParent().(ParserRuleContext)
	} else {
		p.ctx = nil
	}
}

func (p *BaseParser) EnterOuterAlt(localctx ParserRuleContext, altNum int) {
	localctx.SetAltNumber(altNum)
	// if we have Newlocalctx, make sure we replace existing ctx
	// that is previous child of parse tree
	if p.BuildParseTrees && p.ctx != localctx {
		if p.ctx.GetParent() != nil {
			p.ctx.GetParent().(ParserRuleContext).RemoveLastChild()
			p.ctx.GetParent().(ParserRuleContext).AddChild(localctx)
		}
	}
	p.ctx = localctx
}

// Get the precedence level for the top-most precedence rule.
//
// @return The precedence level for the top-most precedence rule, or -1 if
// the parser context is not nested within a precedence rule.

func (p *BaseParser) GetPrecedence() int {
	if len(p.precedenceStack) == 0 {
		return -1
	}

	return p.precedenceStack[len(p.precedenceStack)-1]
}

func (p *BaseParser) EnterRecursionRule(localctx ParserRuleContext, state, ruleIndex, precedence int) {
	p.SetState(state)
	p.precedenceStack.Push(precedence)
	p.ctx = localctx
	p.ctx.SetStart(p.input.LT(1))
	if p.parseListeners != nil {
		p.TriggerEnterRuleEvent() // simulates rule entry for
		// left-recursive rules
	}
}

//
// Like {@link //EnterRule} but for recursive rules.

func (p *BaseParser) PushNewRecursionContext(localctx ParserRuleContext, state, ruleIndex int) {
	previous := p.ctx
	previous.SetParent(localctx)
	previous.SetInvokingState(state)
	previous.SetStop(p.input.LT(-1))

	p.ctx = localctx
	p.ctx.SetStart(previous.GetStart())
	if p.BuildParseTrees {
		p.ctx.AddChild(previous)
	}
	if p.parseListeners != nil {
		p.TriggerEnterRuleEvent() // simulates rule entry for
		// left-recursive rules
	}
}

func (p *BaseParser) UnrollRecursionContexts(parentCtx ParserRuleContext) {
	p.precedenceStack.Pop()
	p.ctx.SetStop(p.input.LT(-1))
	retCtx := p.ctx // save current ctx (return value)
	// unroll so ctx is as it was before call to recursive method
	if p.parseListeners != nil {
		for p.ctx != parentCtx {
			p.TriggerExitRuleEvent()
			p.ctx = p.ctx.GetParent().(ParserRuleContext)
		}
	} else {
		p.ctx = parentCtx
	}
	// hook into tree
	retCtx.SetParent(parentCtx)
	if p.BuildParseTrees && parentCtx != nil {
		// add return ctx into invoking rule's tree
		parentCtx.AddChild(retCtx)
	}
}

func (p *BaseParser) GetInvokingContext(ruleIndex int) ParserRuleContext {
	ctx := p.ctx
	for ctx != nil {
		if ctx.GetRuleIndex() == ruleIndex {
			return ctx
		}
		ctx = ctx.GetParent().(ParserRuleContext)
	}
	return nil
}

func (p *BaseParser) Precpred(localctx RuleContext, precedence int) bool {
	return precedence >= p.precedenceStack[len(p.precedenceStack)-1]
}

func (p *BaseParser) inContext(context ParserRuleContext) bool {
	// TODO: useful in parser?
	return false
}

//
// Checks whether or not {@code symbol} can follow the current state in the
// ATN. The behavior of p.method is equivalent to the following, but is
// implemented such that the complete context-sensitive follow set does not
// need to be explicitly constructed.
//
// <pre>
// return getExpectedTokens().contains(symbol)
// </pre>
//
// @param symbol the symbol type to check
// @return {@code true} if {@code symbol} can follow the current state in
// the ATN, otherwise {@code false}.

func (p *BaseParser) IsExpectedToken(symbol int) bool {
	atn := p.Interpreter.atn
	ctx := p.ctx
	s := atn.states[p.state]
	following := atn.NextTokens(s, nil)
	if following.contains(symbol) {
		return true
	}
	if !following.contains(TokenEpsilon) {
		return false
	}
	for ctx != nil && ctx.GetInvokingState() >= 0 && following.contains(TokenEpsilon) {
		invokingState := atn.states[ctx.GetInvokingState()]
		rt := invokingState.GetTransitions()[0]
		following = atn.NextTokens(rt.(*RuleTransition).followState, nil)
		if following.contains(symbol) {
			return true
		}
		ctx = ctx.GetParent().(ParserRuleContext)
	}
	if following.contains(TokenEpsilon) && symbol == TokenEOF {
		return true
	}

	return false
}

// Computes the set of input symbols which could follow the current parser
// state and context, as given by {@link //GetState} and {@link //GetContext},
// respectively.
//
// @see ATN//getExpectedTokens(int, RuleContext)
//
func (p *BaseParser) GetExpectedTokens() *IntervalSet {
	return p.Interpreter.atn.getExpectedTokens(p.state, p.ctx)
}

func (p *BaseParser) GetExpectedTokensWithinCurrentRule() *IntervalSet {
	atn := p.Interpreter.atn
	s := atn.states[p.state]
	return atn.NextTokens(s, nil)
}

// Get a rule's index (i.e., {@code RULE_ruleName} field) or -1 if not found.//
func (p *BaseParser) GetRuleIndex(ruleName string) int {
	var ruleIndex, ok = p.GetRuleIndexMap()[ruleName]
	if ok {
		return ruleIndex
	}

	return -1
}

// Return List&ltString&gt of the rule names in your parser instance
// leading up to a call to the current rule. You could override if
// you want more details such as the file/line info of where
// in the ATN a rule is invoked.
//
// this very useful for error messages.

func (p *BaseParser) GetRuleInvocationStack(c ParserRuleContext) []string {
	if c == nil {
		c = p.ctx
	}
	stack := make([]string, 0)
	for c != nil {
		// compute what follows who invoked us
		ruleIndex := c.GetRuleIndex()
		if ruleIndex < 0 {
			stack = append(stack, "n/a")
		} else {
			stack = append(stack, p.GetRuleNames()[ruleIndex])
		}

		vp := c.GetParent()

		if vp == nil {
			break
		}

		c = vp.(ParserRuleContext)
	}
	return stack
}

// For debugging and other purposes.//
func (p *BaseParser) GetDFAStrings() string {
	return fmt.Sprint(p.Interpreter.decisionToDFA)
}

// For debugging and other purposes.//
func (p *BaseParser) DumpDFA() {
	seenOne := false
	for _, dfa := range p.Interpreter.decisionToDFA {
		if dfa.numStates() > 0 {
			if seenOne {
				fmt.Println()
			}
			fmt.Println("Decision " + strconv.Itoa(dfa.decision) + ":")
			fmt.Print(dfa.String(p.LiteralNames, p.SymbolicNames))
			seenOne = true
		}
	}
}

func (p *BaseParser) GetSourceName() string {
	return p.GrammarFileName
}

// During a parse is sometimes useful to listen in on the rule entry and exit
// events as well as token Matches. p.is for quick and dirty debugging.
//
func (p *BaseParser) SetTrace(trace *TraceListener) {
	if trace == nil {
		p.RemoveParseListener(p.tracer)
		p.tracer = nil
	} else {
		if p.tracer != nil {
			p.RemoveParseListener(p.tracer)
		}
		p.tracer = NewTraceListener(p)
		p.AddParseListener(p.tracer)
	}
}
