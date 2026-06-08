// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"reflect"
	"strconv"
)

type ParserRuleContext interface {
	RuleContext

	SetException(RecognitionException)

	AddTokenNode(token Token) *TerminalNodeImpl
	AddErrorNode(badToken Token) *ErrorNodeImpl

	EnterRule(listener ParseTreeListener)
	ExitRule(listener ParseTreeListener)

	SetStart(Token)
	GetStart() Token

	SetStop(Token)
	GetStop() Token

	AddChild(child RuleContext) RuleContext
	RemoveLastChild()
}

type BaseParserRuleContext struct {
	parentCtx     RuleContext
	invokingState int
	RuleIndex     int

	start, stop Token
	exception   RecognitionException
	children    []Tree
}

func NewBaseParserRuleContext(parent ParserRuleContext, invokingStateNumber int) *BaseParserRuleContext {
	prc := new(BaseParserRuleContext)
	InitBaseParserRuleContext(prc, parent, invokingStateNumber)
	return prc
}

func InitBaseParserRuleContext(prc *BaseParserRuleContext, parent ParserRuleContext, invokingStateNumber int) {
	// What context invoked b rule?
	prc.parentCtx = parent

	// What state invoked the rule associated with b context?
	// The "return address" is the followState of invokingState
	// If parent is nil, b should be -1.
	if parent == nil {
		prc.invokingState = -1
	} else {
		prc.invokingState = invokingStateNumber
	}

	prc.RuleIndex = -1
	// * If we are debugging or building a parse tree for a Visitor,
	// we need to track all of the tokens and rule invocations associated
	// with prc rule's context. This is empty for parsing w/o tree constr.
	// operation because we don't the need to track the details about
	// how we parse prc rule.
	// /
	prc.children = nil
	prc.start = nil
	prc.stop = nil
	// The exception that forced prc rule to return. If the rule successfully
	// completed, prc is {@code nil}.
	prc.exception = nil
}

func (prc *BaseParserRuleContext) SetException(e RecognitionException) {
	prc.exception = e
}

func (prc *BaseParserRuleContext) GetChildren() []Tree {
	return prc.children
}

func (prc *BaseParserRuleContext) CopyFrom(ctx *BaseParserRuleContext) {
	// from RuleContext
	prc.parentCtx = ctx.parentCtx
	prc.invokingState = ctx.invokingState
	prc.children = nil
	prc.start = ctx.start
	prc.stop = ctx.stop
}

func (prc *BaseParserRuleContext) GetText() string {
	if prc.GetChildCount() == 0 {
		return ""
	}

	var s string
	for _, child := range prc.children {
		s += child.(ParseTree).GetText()
	}

	return s
}

// EnterRule is called when any rule is entered.
func (prc *BaseParserRuleContext) EnterRule(_ ParseTreeListener) {
}

// ExitRule is called when any rule is exited.
func (prc *BaseParserRuleContext) ExitRule(_ ParseTreeListener) {
}

// * Does not set parent link other add methods do that
func (prc *BaseParserRuleContext) addTerminalNodeChild(child TerminalNode) TerminalNode {
	if prc.children == nil {
		prc.children = make([]Tree, 0)
	}
	if child == nil {
		panic("Child may not be null")
	}
	prc.children = append(prc.children, child)
	return child
}

func (prc *BaseParserRuleContext) AddChild(child RuleContext) RuleContext {
	if prc.children == nil {
		prc.children = make([]Tree, 0)
	}
	if child == nil {
		panic("Child may not be null")
	}
	prc.children = append(prc.children, child)
	return child
}

// RemoveLastChild is used by [EnterOuterAlt] to toss out a [RuleContext] previously added as
// we entered a rule. If we have a label, we will need to remove
// the generic ruleContext object.
func (prc *BaseParserRuleContext) RemoveLastChild() {
	if prc.children != nil && len(prc.children) > 0 {
		prc.children = prc.children[0 : len(prc.children)-1]
	}
}

func (prc *BaseParserRuleContext) AddTokenNode(token Token) *TerminalNodeImpl {

	node := NewTerminalNodeImpl(token)
	prc.addTerminalNodeChild(node)
	node.parentCtx = prc
	return node

}

func (prc *BaseParserRuleContext) AddErrorNode(badToken Token) *ErrorNodeImpl {
	node := NewErrorNodeImpl(badToken)
	prc.addTerminalNodeChild(node)
	node.parentCtx = prc
	return node
}

func (prc *BaseParserRuleContext) GetChild(i int) Tree {
	if prc.children != nil && len(prc.children) >= i {
		return prc.children[i]
	}

	return nil
}

func (prc *BaseParserRuleContext) GetChildOfType(i int, childType reflect.Type) RuleContext {
	if childType == nil {
		return prc.GetChild(i).(RuleContext)
	}

	for j := 0; j < len(prc.children); j++ {
		child := prc.children[j]
		if reflect.TypeOf(child) == childType {
			if i == 0 {
				return child.(RuleContext)
			}

			i--
		}
	}

	return nil
}

func (prc *BaseParserRuleContext) ToStringTree(ruleNames []string, recog Recognizer) string {
	return TreesStringTree(prc, ruleNames, recog)
}

func (prc *BaseParserRuleContext) GetRuleContext() RuleContext {
	return prc
}

func (prc *BaseParserRuleContext) Accept(visitor ParseTreeVisitor) interface{} {
	return visitor.VisitChildren(prc)
}

func (prc *BaseParserRuleContext) SetStart(t Token) {
	prc.start = t
}

func (prc *BaseParserRuleContext) GetStart() Token {
	return prc.start
}

func (prc *BaseParserRuleContext) SetStop(t Token) {
	prc.stop = t
}

func (prc *BaseParserRuleContext) GetStop() Token {
	return prc.stop
}

func (prc *BaseParserRuleContext) GetToken(ttype int, i int) TerminalNode {

	for j := 0; j < len(prc.children); j++ {
		child := prc.children[j]
		if c2, ok := child.(TerminalNode); ok {
			if c2.GetSymbol().GetTokenType() == ttype {
				if i == 0 {
					return c2
				}

				i--
			}
		}
	}
	return nil
}

func (prc *BaseParserRuleContext) GetTokens(ttype int) []TerminalNode {
	if prc.children == nil {
		return make([]TerminalNode, 0)
	}

	tokens := make([]TerminalNode, 0)

	for j := 0; j < len(prc.children); j++ {
		child := prc.children[j]
		if tchild, ok := child.(TerminalNode); ok {
			if tchild.GetSymbol().GetTokenType() == ttype {
				tokens = append(tokens, tchild)
			}
		}
	}

	return tokens
}

func (prc *BaseParserRuleContext) GetPayload() interface{} {
	return prc
}

func (prc *BaseParserRuleContext) getChild(ctxType reflect.Type, i int) RuleContext {
	if prc.children == nil || i < 0 || i >= len(prc.children) {
		return nil
	}

	j := -1 // what element have we found with ctxType?
	for _, o := range prc.children {

		childType := reflect.TypeOf(o)

		if childType.Implements(ctxType) {
			j++
			if j == i {
				return o.(RuleContext)
			}
		}
	}
	return nil
}

// Go lacks generics, so it's not possible for us to return the child with the correct type, but we do
// check for convertibility

func (prc *BaseParserRuleContext) GetTypedRuleContext(ctxType reflect.Type, i int) RuleContext {
	return prc.getChild(ctxType, i)
}

func (prc *BaseParserRuleContext) GetTypedRuleContexts(ctxType reflect.Type) []RuleContext {
	if prc.children == nil {
		return make([]RuleContext, 0)
	}

	contexts := make([]RuleContext, 0)

	for _, child := range prc.children {
		childType := reflect.TypeOf(child)

		if childType.ConvertibleTo(ctxType) {
			contexts = append(contexts, child.(RuleContext))
		}
	}
	return contexts
}

func (prc *BaseParserRuleContext) GetChildCount() int {
	if prc.children == nil {
		return 0
	}

	return len(prc.children)
}

func (prc *BaseParserRuleContext) GetSourceInterval() Interval {
	if prc.start == nil || prc.stop == nil {
		return TreeInvalidInterval
	}

	return NewInterval(prc.start.GetTokenIndex(), prc.stop.GetTokenIndex())
}

//need to manage circular dependencies, so export now

// Print out a whole tree, not just a node, in LISP format
// (root child1 .. childN). Print just a node if b is a leaf.
//

func (prc *BaseParserRuleContext) String(ruleNames []string, stop RuleContext) string {

	var p ParserRuleContext = prc
	s := "["
	for p != nil && p != stop {
		if ruleNames == nil {
			if !p.IsEmpty() {
				s += strconv.Itoa(p.GetInvokingState())
			}
		} else {
			ri := p.GetRuleIndex()
			var ruleName string
			if ri >= 0 && ri < len(ruleNames) {
				ruleName = ruleNames[ri]
			} else {
				ruleName = strconv.Itoa(ri)
			}
			s += ruleName
		}
		if p.GetParent() != nil && (ruleNames != nil || !p.GetParent().(ParserRuleContext).IsEmpty()) {
			s += " "
		}
		pi := p.GetParent()
		if pi != nil {
			p = pi.(ParserRuleContext)
		} else {
			p = nil
		}
	}
	s += "]"
	return s
}

func (prc *BaseParserRuleContext) SetParent(v Tree) {
	if v == nil {
		prc.parentCtx = nil
	} else {
		prc.parentCtx = v.(RuleContext)
	}
}

func (prc *BaseParserRuleContext) GetInvokingState() int {
	return prc.invokingState
}

func (prc *BaseParserRuleContext) SetInvokingState(t int) {
	prc.invokingState = t
}

func (prc *BaseParserRuleContext) GetRuleIndex() int {
	return prc.RuleIndex
}

func (prc *BaseParserRuleContext) GetAltNumber() int {
	return ATNInvalidAltNumber
}

func (prc *BaseParserRuleContext) SetAltNumber(_ int) {}

// IsEmpty returns true if the context of b is empty.
//
// A context is empty if there is no invoking state, meaning nobody calls
// current context.
func (prc *BaseParserRuleContext) IsEmpty() bool {
	return prc.invokingState == -1
}

// GetParent returns the combined text of all child nodes. This method only considers
// tokens which have been added to the parse tree.
//
// Since tokens on hidden channels (e.g. whitespace or comments) are not
// added to the parse trees, they will not appear in the output of this
// method.
func (prc *BaseParserRuleContext) GetParent() Tree {
	return prc.parentCtx
}

var ParserRuleContextEmpty = NewBaseParserRuleContext(nil, -1)

type InterpreterRuleContext interface {
	ParserRuleContext
}

type BaseInterpreterRuleContext struct {
	*BaseParserRuleContext
}

//goland:noinspection GoUnusedExportedFunction
func NewBaseInterpreterRuleContext(parent BaseInterpreterRuleContext, invokingStateNumber, ruleIndex int) *BaseInterpreterRuleContext {

	prc := new(BaseInterpreterRuleContext)

	prc.BaseParserRuleContext = NewBaseParserRuleContext(parent, invokingStateNumber)

	prc.RuleIndex = ruleIndex

	return prc
}
