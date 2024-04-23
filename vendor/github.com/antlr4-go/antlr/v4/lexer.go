// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
	"strconv"
)

// A lexer is recognizer that draws input symbols from a character stream.
//  lexer grammars result in a subclass of this object. A Lexer object
//  uses simplified Match() and error recovery mechanisms in the interest
//  of speed.
///

type Lexer interface {
	TokenSource
	Recognizer

	Emit() Token

	SetChannel(int)
	PushMode(int)
	PopMode() int
	SetType(int)
	SetMode(int)
}

type BaseLexer struct {
	*BaseRecognizer

	Interpreter         ILexerATNSimulator
	TokenStartCharIndex int
	TokenStartLine      int
	TokenStartColumn    int
	ActionType          int
	Virt                Lexer // The most derived lexer implementation. Allows virtual method calls.

	input                  CharStream
	factory                TokenFactory
	tokenFactorySourcePair *TokenSourceCharStreamPair
	token                  Token
	hitEOF                 bool
	channel                int
	thetype                int
	modeStack              IntStack
	mode                   int
	text                   string
}

func NewBaseLexer(input CharStream) *BaseLexer {

	lexer := new(BaseLexer)

	lexer.BaseRecognizer = NewBaseRecognizer()

	lexer.input = input
	lexer.factory = CommonTokenFactoryDEFAULT
	lexer.tokenFactorySourcePair = &TokenSourceCharStreamPair{lexer, input}

	lexer.Virt = lexer

	lexer.Interpreter = nil // child classes must populate it

	// The goal of all lexer rules/methods is to create a token object.
	// l is an instance variable as multiple rules may collaborate to
	// create a single token. NextToken will return l object after
	// Matching lexer rule(s). If you subclass to allow multiple token
	// emissions, then set l to the last token to be Matched or
	// something non nil so that the auto token emit mechanism will not
	// emit another token.
	lexer.token = nil

	// What character index in the stream did the current token start at?
	// Needed, for example, to get the text for current token. Set at
	// the start of NextToken.
	lexer.TokenStartCharIndex = -1

	// The line on which the first character of the token resides///
	lexer.TokenStartLine = -1

	// The character position of first character within the line///
	lexer.TokenStartColumn = -1

	// Once we see EOF on char stream, next token will be EOF.
	// If you have DONE : EOF  then you see DONE EOF.
	lexer.hitEOF = false

	// The channel number for the current token///
	lexer.channel = TokenDefaultChannel

	// The token type for the current token///
	lexer.thetype = TokenInvalidType

	lexer.modeStack = make([]int, 0)
	lexer.mode = LexerDefaultMode

	// You can set the text for the current token to override what is in
	// the input char buffer. Use setText() or can set l instance var.
	// /
	lexer.text = ""

	return lexer
}

const (
	LexerDefaultMode = 0
	LexerMore        = -2
	LexerSkip        = -3
)

//goland:noinspection GoUnusedConst
const (
	LexerDefaultTokenChannel = TokenDefaultChannel
	LexerHidden              = TokenHiddenChannel
	LexerMinCharValue        = 0x0000
	LexerMaxCharValue        = 0x10FFFF
)

func (b *BaseLexer) Reset() {
	// wack Lexer state variables
	if b.input != nil {
		b.input.Seek(0) // rewind the input
	}
	b.token = nil
	b.thetype = TokenInvalidType
	b.channel = TokenDefaultChannel
	b.TokenStartCharIndex = -1
	b.TokenStartColumn = -1
	b.TokenStartLine = -1
	b.text = ""

	b.hitEOF = false
	b.mode = LexerDefaultMode
	b.modeStack = make([]int, 0)

	b.Interpreter.reset()
}

func (b *BaseLexer) GetInterpreter() ILexerATNSimulator {
	return b.Interpreter
}

func (b *BaseLexer) GetInputStream() CharStream {
	return b.input
}

func (b *BaseLexer) GetSourceName() string {
	return b.GrammarFileName
}

func (b *BaseLexer) SetChannel(v int) {
	b.channel = v
}

func (b *BaseLexer) GetTokenFactory() TokenFactory {
	return b.factory
}

func (b *BaseLexer) setTokenFactory(f TokenFactory) {
	b.factory = f
}

func (b *BaseLexer) safeMatch() (ret int) {
	defer func() {
		if e := recover(); e != nil {
			if re, ok := e.(RecognitionException); ok {
				b.notifyListeners(re) // Report error
				b.Recover(re)
				ret = LexerSkip // default
			}
		}
	}()

	return b.Interpreter.Match(b.input, b.mode)
}

// NextToken returns a token from the lexer input source i.e., Match a token on the source char stream.
func (b *BaseLexer) NextToken() Token {
	if b.input == nil {
		panic("NextToken requires a non-nil input stream.")
	}

	tokenStartMarker := b.input.Mark()

	// previously in finally block
	defer func() {
		// make sure we release marker after Match or
		// unbuffered char stream will keep buffering
		b.input.Release(tokenStartMarker)
	}()

	for {
		if b.hitEOF {
			b.EmitEOF()
			return b.token
		}
		b.token = nil
		b.channel = TokenDefaultChannel
		b.TokenStartCharIndex = b.input.Index()
		b.TokenStartColumn = b.Interpreter.GetCharPositionInLine()
		b.TokenStartLine = b.Interpreter.GetLine()
		b.text = ""
		continueOuter := false
		for {
			b.thetype = TokenInvalidType

			ttype := b.safeMatch()

			if b.input.LA(1) == TokenEOF {
				b.hitEOF = true
			}
			if b.thetype == TokenInvalidType {
				b.thetype = ttype
			}
			if b.thetype == LexerSkip {
				continueOuter = true
				break
			}
			if b.thetype != LexerMore {
				break
			}
		}

		if continueOuter {
			continue
		}
		if b.token == nil {
			b.Virt.Emit()
		}
		return b.token
	}
}

// Skip instructs the lexer to Skip creating a token for current lexer rule
// and look for another token. [NextToken] knows to keep looking when
// a lexer rule finishes with token set to [SKIPTOKEN]. Recall that
// if token==nil at end of any token rule, it creates one for you
// and emits it.
func (b *BaseLexer) Skip() {
	b.thetype = LexerSkip
}

func (b *BaseLexer) More() {
	b.thetype = LexerMore
}

// SetMode changes the lexer to a new mode. The lexer will use this mode from hereon in and the rules for that mode
// will be in force.
func (b *BaseLexer) SetMode(m int) {
	b.mode = m
}

// PushMode saves the current lexer mode so that it can be restored later. See [PopMode], then sets the
// current lexer mode to the supplied mode m.
func (b *BaseLexer) PushMode(m int) {
	if runtimeConfig.lexerATNSimulatorDebug {
		fmt.Println("pushMode " + strconv.Itoa(m))
	}
	b.modeStack.Push(b.mode)
	b.mode = m
}

// PopMode restores the lexer mode saved by a call to [PushMode]. It is a panic error if there is no saved mode to
// return to.
func (b *BaseLexer) PopMode() int {
	if len(b.modeStack) == 0 {
		panic("Empty Stack")
	}
	if runtimeConfig.lexerATNSimulatorDebug {
		fmt.Println("popMode back to " + fmt.Sprint(b.modeStack[0:len(b.modeStack)-1]))
	}
	i, _ := b.modeStack.Pop()
	b.mode = i
	return b.mode
}

func (b *BaseLexer) inputStream() CharStream {
	return b.input
}

// SetInputStream resets the lexer input stream and associated lexer state.
func (b *BaseLexer) SetInputStream(input CharStream) {
	b.input = nil
	b.tokenFactorySourcePair = &TokenSourceCharStreamPair{b, b.input}
	b.Reset()
	b.input = input
	b.tokenFactorySourcePair = &TokenSourceCharStreamPair{b, b.input}
}

func (b *BaseLexer) GetTokenSourceCharStreamPair() *TokenSourceCharStreamPair {
	return b.tokenFactorySourcePair
}

// EmitToken by default does not support multiple emits per [NextToken] invocation
// for efficiency reasons. Subclass and override this func, [NextToken],
// and [GetToken] (to push tokens into a list and pull from that list
// rather than a single variable as this implementation does).
func (b *BaseLexer) EmitToken(token Token) {
	b.token = token
}

// Emit is the standard method called to automatically emit a token at the
// outermost lexical rule. The token object should point into the
// char buffer start..stop. If there is a text override in 'text',
// use that to set the token's text. Override this method to emit
// custom [Token] objects or provide a new factory.
// /
func (b *BaseLexer) Emit() Token {
	t := b.factory.Create(b.tokenFactorySourcePair, b.thetype, b.text, b.channel, b.TokenStartCharIndex, b.GetCharIndex()-1, b.TokenStartLine, b.TokenStartColumn)
	b.EmitToken(t)
	return t
}

// EmitEOF emits an EOF token. By default, this is the last token emitted
func (b *BaseLexer) EmitEOF() Token {
	cpos := b.GetCharPositionInLine()
	lpos := b.GetLine()
	eof := b.factory.Create(b.tokenFactorySourcePair, TokenEOF, "", TokenDefaultChannel, b.input.Index(), b.input.Index()-1, lpos, cpos)
	b.EmitToken(eof)
	return eof
}

// GetCharPositionInLine returns the current position in the current line as far as the lexer is concerned.
func (b *BaseLexer) GetCharPositionInLine() int {
	return b.Interpreter.GetCharPositionInLine()
}

func (b *BaseLexer) GetLine() int {
	return b.Interpreter.GetLine()
}

func (b *BaseLexer) GetType() int {
	return b.thetype
}

func (b *BaseLexer) SetType(t int) {
	b.thetype = t
}

// GetCharIndex returns the index of the current character of lookahead
func (b *BaseLexer) GetCharIndex() int {
	return b.input.Index()
}

// GetText returns the text Matched so far for the current token or any text override.
func (b *BaseLexer) GetText() string {
	if b.text != "" {
		return b.text
	}

	return b.Interpreter.GetText(b.input)
}

// SetText sets the complete text of this token; it wipes any previous changes to the text.
func (b *BaseLexer) SetText(text string) {
	b.text = text
}

// GetATN returns the ATN used by the lexer.
func (b *BaseLexer) GetATN() *ATN {
	return b.Interpreter.ATN()
}

// GetAllTokens returns a list of all [Token] objects in input char stream.
// Forces a load of all tokens that can be made from the input char stream.
//
// Does not include EOF token.
func (b *BaseLexer) GetAllTokens() []Token {
	vl := b.Virt
	tokens := make([]Token, 0)
	t := vl.NextToken()
	for t.GetTokenType() != TokenEOF {
		tokens = append(tokens, t)
		t = vl.NextToken()
	}
	return tokens
}

func (b *BaseLexer) notifyListeners(e RecognitionException) {
	start := b.TokenStartCharIndex
	stop := b.input.Index()
	text := b.input.GetTextFromInterval(NewInterval(start, stop))
	msg := "token recognition error at: '" + text + "'"
	listener := b.GetErrorListenerDispatch()
	listener.SyntaxError(b, nil, b.TokenStartLine, b.TokenStartColumn, msg, e)
}

func (b *BaseLexer) getErrorDisplayForChar(c rune) string {
	if c == TokenEOF {
		return "<EOF>"
	} else if c == '\n' {
		return "\\n"
	} else if c == '\t' {
		return "\\t"
	} else if c == '\r' {
		return "\\r"
	} else {
		return string(c)
	}
}

func (b *BaseLexer) getCharErrorDisplay(c rune) string {
	return "'" + b.getErrorDisplayForChar(c) + "'"
}

// Recover can normally Match any char in its vocabulary after Matching
// a token, so here we do the easy thing and just kill a character and hope
// it all works out. You can instead use the rule invocation stack
// to do sophisticated error recovery if you are in a fragment rule.
//
// In general, lexers should not need to recover and should have rules that cover any eventuality, such as
// a character that makes no sense to the recognizer.
func (b *BaseLexer) Recover(re RecognitionException) {
	if b.input.LA(1) != TokenEOF {
		if _, ok := re.(*LexerNoViableAltException); ok {
			// Skip a char and try again
			b.Interpreter.Consume(b.input)
		} else {
			// TODO: Do we lose character or line position information?
			b.input.Consume()
		}
	}
}
