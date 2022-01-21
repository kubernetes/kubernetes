// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import "fmt"

/** A set of utility routines useful for all kinds of ANTLR trees. */

// Print out a whole tree in LISP form. {@link //getNodeText} is used on the
//  node payloads to get the text for the nodes.  Detect
//  parse trees and extract data appropriately.
func TreesStringTree(tree Tree, ruleNames []string, recog Recognizer) string {

	if recog != nil {
		ruleNames = recog.GetRuleNames()
	}

	s := TreesGetNodeText(tree, ruleNames, nil)

	s = EscapeWhitespace(s, false)
	c := tree.GetChildCount()
	if c == 0 {
		return s
	}
	res := "(" + s + " "
	if c > 0 {
		s = TreesStringTree(tree.GetChild(0), ruleNames, nil)
		res += s
	}
	for i := 1; i < c; i++ {
		s = TreesStringTree(tree.GetChild(i), ruleNames, nil)
		res += (" " + s)
	}
	res += ")"
	return res
}

func TreesGetNodeText(t Tree, ruleNames []string, recog Parser) string {
	if recog != nil {
		ruleNames = recog.GetRuleNames()
	}

	if ruleNames != nil {
		switch t2 := t.(type) {
		case RuleNode:
			t3 := t2.GetRuleContext()
			altNumber := t3.GetAltNumber()

			if altNumber != ATNInvalidAltNumber {
				return fmt.Sprintf("%s:%d", ruleNames[t3.GetRuleIndex()], altNumber)
			}
			return ruleNames[t3.GetRuleIndex()]
		case ErrorNode:
			return fmt.Sprint(t2)
		case TerminalNode:
			if t2.GetSymbol() != nil {
				return t2.GetSymbol().GetText()
			}
		}
	}

	// no recog for rule names
	payload := t.GetPayload()
	if p2, ok := payload.(Token); ok {
		return p2.GetText()
	}

	return fmt.Sprint(t.GetPayload())
}

// Return ordered list of all children of this node
func TreesGetChildren(t Tree) []Tree {
	list := make([]Tree, 0)
	for i := 0; i < t.GetChildCount(); i++ {
		list = append(list, t.GetChild(i))
	}
	return list
}

// Return a list of all ancestors of this node.  The first node of
//  list is the root and the last is the parent of this node.
//
func TreesgetAncestors(t Tree) []Tree {
	ancestors := make([]Tree, 0)
	t = t.GetParent()
	for t != nil {
		f := []Tree{t}
		ancestors = append(f, ancestors...)
		t = t.GetParent()
	}
	return ancestors
}

func TreesFindAllTokenNodes(t ParseTree, ttype int) []ParseTree {
	return TreesfindAllNodes(t, ttype, true)
}

func TreesfindAllRuleNodes(t ParseTree, ruleIndex int) []ParseTree {
	return TreesfindAllNodes(t, ruleIndex, false)
}

func TreesfindAllNodes(t ParseTree, index int, findTokens bool) []ParseTree {
	nodes := make([]ParseTree, 0)
	treesFindAllNodes(t, index, findTokens, &nodes)
	return nodes
}

func treesFindAllNodes(t ParseTree, index int, findTokens bool, nodes *[]ParseTree) {
	// check this node (the root) first

	t2, ok := t.(TerminalNode)
	t3, ok2 := t.(ParserRuleContext)

	if findTokens && ok {
		if t2.GetSymbol().GetTokenType() == index {
			*nodes = append(*nodes, t2)
		}
	} else if !findTokens && ok2 {
		if t3.GetRuleIndex() == index {
			*nodes = append(*nodes, t3)
		}
	}
	// check children
	for i := 0; i < t.GetChildCount(); i++ {
		treesFindAllNodes(t.GetChild(i).(ParseTree), index, findTokens, nodes)
	}
}

func TreesDescendants(t ParseTree) []ParseTree {
	nodes := []ParseTree{t}
	for i := 0; i < t.GetChildCount(); i++ {
		nodes = append(nodes, TreesDescendants(t.GetChild(i).(ParseTree))...)
	}
	return nodes
}
