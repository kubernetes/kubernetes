// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"fmt"
	"strings"

	"strconv"
)

type Recognizer interface {
	GetLiteralNames() []string
	GetSymbolicNames() []string
	GetRuleNames() []string

	Sempred(RuleContext, int, int) bool
	Precpred(RuleContext, int) bool

	GetState() int
	SetState(int)
	Action(RuleContext, int, int)
	AddErrorListener(ErrorListener)
	RemoveErrorListeners()
	GetATN() *ATN
	GetErrorListenerDispatch() ErrorListener
	HasError() bool
	GetError() RecognitionException
	SetError(RecognitionException)
}

type BaseRecognizer struct {
	listeners []ErrorListener
	state     int

	RuleNames       []string
	LiteralNames    []string
	SymbolicNames   []string
	GrammarFileName string
	SynErr          RecognitionException
}

func NewBaseRecognizer() *BaseRecognizer {
	rec := new(BaseRecognizer)
	rec.listeners = []ErrorListener{ConsoleErrorListenerINSTANCE}
	rec.state = -1
	return rec
}

//goland:noinspection GoUnusedGlobalVariable
var tokenTypeMapCache = make(map[string]int)

//goland:noinspection GoUnusedGlobalVariable
var ruleIndexMapCache = make(map[string]int)

func (b *BaseRecognizer) checkVersion(toolVersion string) {
	runtimeVersion := "4.13.1"
	if runtimeVersion != toolVersion {
		fmt.Println("ANTLR runtime and generated code versions disagree: " + runtimeVersion + "!=" + toolVersion)
	}
}

func (b *BaseRecognizer) SetError(err RecognitionException) {
	b.SynErr = err
}

func (b *BaseRecognizer) HasError() bool {
	return b.SynErr != nil
}

func (b *BaseRecognizer) GetError() RecognitionException {
	return b.SynErr
}

func (b *BaseRecognizer) Action(_ RuleContext, _, _ int) {
	panic("action not implemented on Recognizer!")
}

func (b *BaseRecognizer) AddErrorListener(listener ErrorListener) {
	b.listeners = append(b.listeners, listener)
}

func (b *BaseRecognizer) RemoveErrorListeners() {
	b.listeners = make([]ErrorListener, 0)
}

func (b *BaseRecognizer) GetRuleNames() []string {
	return b.RuleNames
}

func (b *BaseRecognizer) GetTokenNames() []string {
	return b.LiteralNames
}

func (b *BaseRecognizer) GetSymbolicNames() []string {
	return b.SymbolicNames
}

func (b *BaseRecognizer) GetLiteralNames() []string {
	return b.LiteralNames
}

func (b *BaseRecognizer) GetState() int {
	return b.state
}

func (b *BaseRecognizer) SetState(v int) {
	b.state = v
}

//func (b *Recognizer) GetTokenTypeMap() {
//    var tokenNames = b.GetTokenNames()
//    if (tokenNames==nil) {
//        panic("The current recognizer does not provide a list of token names.")
//    }
//    var result = tokenTypeMapCache[tokenNames]
//    if(result==nil) {
//        result = tokenNames.reduce(function(o, k, i) { o[k] = i })
//        result.EOF = TokenEOF
//        tokenTypeMapCache[tokenNames] = result
//    }
//    return result
//}

// GetRuleIndexMap Get a map from rule names to rule indexes.
//
// Used for XPath and tree pattern compilation.
//
// TODO: JI This is not yet implemented in the Go runtime. Maybe not needed.
func (b *BaseRecognizer) GetRuleIndexMap() map[string]int {

	panic("Method not defined!")
	//    var ruleNames = b.GetRuleNames()
	//    if (ruleNames==nil) {
	//        panic("The current recognizer does not provide a list of rule names.")
	//    }
	//
	//    var result = ruleIndexMapCache[ruleNames]
	//    if(result==nil) {
	//        result = ruleNames.reduce(function(o, k, i) { o[k] = i })
	//        ruleIndexMapCache[ruleNames] = result
	//    }
	//    return result
}

// GetTokenType get the token type based upon its name
func (b *BaseRecognizer) GetTokenType(_ string) int {
	panic("Method not defined!")
	//    var ttype = b.GetTokenTypeMap()[tokenName]
	//    if (ttype !=nil) {
	//        return ttype
	//    } else {
	//        return TokenInvalidType
	//    }
}

//func (b *Recognizer) GetTokenTypeMap() map[string]int {
//    Vocabulary vocabulary = getVocabulary()
//
//    Synchronized (tokenTypeMapCache) {
//        Map<String, Integer> result = tokenTypeMapCache.Get(vocabulary)
//        if (result == null) {
//            result = new HashMap<String, Integer>()
//            for (int i = 0; i < GetATN().maxTokenType; i++) {
//                String literalName = vocabulary.getLiteralName(i)
//                if (literalName != null) {
//                    result.put(literalName, i)
//                }
//
//                String symbolicName = vocabulary.GetSymbolicName(i)
//                if (symbolicName != null) {
//                    result.put(symbolicName, i)
//                }
//            }
//
//            result.put("EOF", Token.EOF)
//            result = Collections.unmodifiableMap(result)
//            tokenTypeMapCache.put(vocabulary, result)
//        }
//
//        return result
//    }
//}

// GetErrorHeader returns the error header, normally line/character position information.
//
// Can be overridden in sub structs embedding BaseRecognizer.
func (b *BaseRecognizer) GetErrorHeader(e RecognitionException) string {
	line := e.GetOffendingToken().GetLine()
	column := e.GetOffendingToken().GetColumn()
	return "line " + strconv.Itoa(line) + ":" + strconv.Itoa(column)
}

// GetTokenErrorDisplay shows how a token should be displayed in an error message.
//
// The default is to display just the text, but during development you might
// want to have a lot of information spit out.  Override in that case
// to use t.String() (which, for CommonToken, dumps everything about
// the token). This is better than forcing you to override a method in
// your token objects because you don't have to go modify your lexer
// so that it creates a NewJava type.
//
// Deprecated: This method is not called by the ANTLR 4 Runtime. Specific
// implementations of [ANTLRErrorStrategy] may provide a similar
// feature when necessary. For example, see [DefaultErrorStrategy].GetTokenErrorDisplay()
func (b *BaseRecognizer) GetTokenErrorDisplay(t Token) string {
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
	s = strings.Replace(s, "\t", "\\t", -1)
	s = strings.Replace(s, "\n", "\\n", -1)
	s = strings.Replace(s, "\r", "\\r", -1)

	return "'" + s + "'"
}

func (b *BaseRecognizer) GetErrorListenerDispatch() ErrorListener {
	return NewProxyErrorListener(b.listeners)
}

// Sempred embedding structs need to override this if there are sempreds or actions
// that the ATN interpreter needs to execute
func (b *BaseRecognizer) Sempred(_ RuleContext, _ int, _ int) bool {
	return true
}

// Precpred embedding structs need to override this if there are preceding predicates
// that the ATN interpreter needs to execute
func (b *BaseRecognizer) Precpred(_ RuleContext, _ int) bool {
	return true
}
