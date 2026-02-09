// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

// RuleContext is a record of a single rule invocation. It knows
// which context invoked it, if any. If there is no parent context, then
// naturally the invoking state is not valid.  The parent link
// provides a chain upwards from the current rule invocation to the root
// of the invocation tree, forming a stack.
//
// We actually carry no information about the rule associated with this context (except
// when parsing). We keep only the state number of the invoking state from
// the [ATN] submachine that invoked this. Contrast this with the s
// pointer inside [ParserRuleContext] that tracks the current state
// being "executed" for the current rule.
//
// The parent contexts are useful for computing lookahead sets and
// getting error information.
//
// These objects are used during parsing and prediction.
// For the special case of parsers, we use the struct
// [ParserRuleContext], which embeds a RuleContext.
//
// @see ParserRuleContext
type RuleContext interface {
	RuleNode

	GetInvokingState() int
	SetInvokingState(int)

	GetRuleIndex() int
	IsEmpty() bool

	GetAltNumber() int
	SetAltNumber(altNumber int)

	String([]string, RuleContext) string
}
