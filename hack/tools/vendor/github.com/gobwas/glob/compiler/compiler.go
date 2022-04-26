package compiler

// TODO use constructor with all matchers, and to their structs private
// TODO glue multiple Text nodes (like after QuoteMeta)

import (
	"fmt"
	"reflect"

	"github.com/gobwas/glob/match"
	"github.com/gobwas/glob/syntax/ast"
	"github.com/gobwas/glob/util/runes"
)

func optimizeMatcher(matcher match.Matcher) match.Matcher {
	switch m := matcher.(type) {

	case match.Any:
		if len(m.Separators) == 0 {
			return match.NewSuper()
		}

	case match.AnyOf:
		if len(m.Matchers) == 1 {
			return m.Matchers[0]
		}

		return m

	case match.List:
		if m.Not == false && len(m.List) == 1 {
			return match.NewText(string(m.List))
		}

		return m

	case match.BTree:
		m.Left = optimizeMatcher(m.Left)
		m.Right = optimizeMatcher(m.Right)

		r, ok := m.Value.(match.Text)
		if !ok {
			return m
		}

		var (
			leftNil  = m.Left == nil
			rightNil = m.Right == nil
		)
		if leftNil && rightNil {
			return match.NewText(r.Str)
		}

		_, leftSuper := m.Left.(match.Super)
		lp, leftPrefix := m.Left.(match.Prefix)
		la, leftAny := m.Left.(match.Any)

		_, rightSuper := m.Right.(match.Super)
		rs, rightSuffix := m.Right.(match.Suffix)
		ra, rightAny := m.Right.(match.Any)

		switch {
		case leftSuper && rightSuper:
			return match.NewContains(r.Str, false)

		case leftSuper && rightNil:
			return match.NewSuffix(r.Str)

		case rightSuper && leftNil:
			return match.NewPrefix(r.Str)

		case leftNil && rightSuffix:
			return match.NewPrefixSuffix(r.Str, rs.Suffix)

		case rightNil && leftPrefix:
			return match.NewPrefixSuffix(lp.Prefix, r.Str)

		case rightNil && leftAny:
			return match.NewSuffixAny(r.Str, la.Separators)

		case leftNil && rightAny:
			return match.NewPrefixAny(r.Str, ra.Separators)
		}

		return m
	}

	return matcher
}

func compileMatchers(matchers []match.Matcher) (match.Matcher, error) {
	if len(matchers) == 0 {
		return nil, fmt.Errorf("compile error: need at least one matcher")
	}
	if len(matchers) == 1 {
		return matchers[0], nil
	}
	if m := glueMatchers(matchers); m != nil {
		return m, nil
	}

	idx := -1
	maxLen := -1
	var val match.Matcher
	for i, matcher := range matchers {
		if l := matcher.Len(); l != -1 && l >= maxLen {
			maxLen = l
			idx = i
			val = matcher
		}
	}

	if val == nil { // not found matcher with static length
		r, err := compileMatchers(matchers[1:])
		if err != nil {
			return nil, err
		}
		return match.NewBTree(matchers[0], nil, r), nil
	}

	left := matchers[:idx]
	var right []match.Matcher
	if len(matchers) > idx+1 {
		right = matchers[idx+1:]
	}

	var l, r match.Matcher
	var err error
	if len(left) > 0 {
		l, err = compileMatchers(left)
		if err != nil {
			return nil, err
		}
	}

	if len(right) > 0 {
		r, err = compileMatchers(right)
		if err != nil {
			return nil, err
		}
	}

	return match.NewBTree(val, l, r), nil
}

func glueMatchers(matchers []match.Matcher) match.Matcher {
	if m := glueMatchersAsEvery(matchers); m != nil {
		return m
	}
	if m := glueMatchersAsRow(matchers); m != nil {
		return m
	}
	return nil
}

func glueMatchersAsRow(matchers []match.Matcher) match.Matcher {
	if len(matchers) <= 1 {
		return nil
	}

	var (
		c []match.Matcher
		l int
	)
	for _, matcher := range matchers {
		if ml := matcher.Len(); ml == -1 {
			return nil
		} else {
			c = append(c, matcher)
			l += ml
		}
	}
	return match.NewRow(l, c...)
}

func glueMatchersAsEvery(matchers []match.Matcher) match.Matcher {
	if len(matchers) <= 1 {
		return nil
	}

	var (
		hasAny    bool
		hasSuper  bool
		hasSingle bool
		min       int
		separator []rune
	)

	for i, matcher := range matchers {
		var sep []rune

		switch m := matcher.(type) {
		case match.Super:
			sep = []rune{}
			hasSuper = true

		case match.Any:
			sep = m.Separators
			hasAny = true

		case match.Single:
			sep = m.Separators
			hasSingle = true
			min++

		case match.List:
			if !m.Not {
				return nil
			}
			sep = m.List
			hasSingle = true
			min++

		default:
			return nil
		}

		// initialize
		if i == 0 {
			separator = sep
		}

		if runes.Equal(sep, separator) {
			continue
		}

		return nil
	}

	if hasSuper && !hasAny && !hasSingle {
		return match.NewSuper()
	}

	if hasAny && !hasSuper && !hasSingle {
		return match.NewAny(separator)
	}

	if (hasAny || hasSuper) && min > 0 && len(separator) == 0 {
		return match.NewMin(min)
	}

	every := match.NewEveryOf()

	if min > 0 {
		every.Add(match.NewMin(min))

		if !hasAny && !hasSuper {
			every.Add(match.NewMax(min))
		}
	}

	if len(separator) > 0 {
		every.Add(match.NewContains(string(separator), true))
	}

	return every
}

func minimizeMatchers(matchers []match.Matcher) []match.Matcher {
	var done match.Matcher
	var left, right, count int

	for l := 0; l < len(matchers); l++ {
		for r := len(matchers); r > l; r-- {
			if glued := glueMatchers(matchers[l:r]); glued != nil {
				var swap bool

				if done == nil {
					swap = true
				} else {
					cl, gl := done.Len(), glued.Len()
					swap = cl > -1 && gl > -1 && gl > cl
					swap = swap || count < r-l
				}

				if swap {
					done = glued
					left = l
					right = r
					count = r - l
				}
			}
		}
	}

	if done == nil {
		return matchers
	}

	next := append(append([]match.Matcher{}, matchers[:left]...), done)
	if right < len(matchers) {
		next = append(next, matchers[right:]...)
	}

	if len(next) == len(matchers) {
		return next
	}

	return minimizeMatchers(next)
}

// minimizeAnyOf tries to apply some heuristics to minimize number of nodes in given tree
func minimizeTree(tree *ast.Node) *ast.Node {
	switch tree.Kind {
	case ast.KindAnyOf:
		return minimizeTreeAnyOf(tree)
	default:
		return nil
	}
}

// minimizeAnyOf tries to find common children of given node of AnyOf pattern
// it searches for common children from left and from right
// if any common children are found – then it returns new optimized ast tree
// else it returns nil
func minimizeTreeAnyOf(tree *ast.Node) *ast.Node {
	if !areOfSameKind(tree.Children, ast.KindPattern) {
		return nil
	}

	commonLeft, commonRight := commonChildren(tree.Children)
	commonLeftCount, commonRightCount := len(commonLeft), len(commonRight)
	if commonLeftCount == 0 && commonRightCount == 0 { // there are no common parts
		return nil
	}

	var result []*ast.Node
	if commonLeftCount > 0 {
		result = append(result, ast.NewNode(ast.KindPattern, nil, commonLeft...))
	}

	var anyOf []*ast.Node
	for _, child := range tree.Children {
		reuse := child.Children[commonLeftCount : len(child.Children)-commonRightCount]
		var node *ast.Node
		if len(reuse) == 0 {
			// this pattern is completely reduced by commonLeft and commonRight patterns
			// so it become nothing
			node = ast.NewNode(ast.KindNothing, nil)
		} else {
			node = ast.NewNode(ast.KindPattern, nil, reuse...)
		}
		anyOf = appendIfUnique(anyOf, node)
	}
	switch {
	case len(anyOf) == 1 && anyOf[0].Kind != ast.KindNothing:
		result = append(result, anyOf[0])
	case len(anyOf) > 1:
		result = append(result, ast.NewNode(ast.KindAnyOf, nil, anyOf...))
	}

	if commonRightCount > 0 {
		result = append(result, ast.NewNode(ast.KindPattern, nil, commonRight...))
	}

	return ast.NewNode(ast.KindPattern, nil, result...)
}

func commonChildren(nodes []*ast.Node) (commonLeft, commonRight []*ast.Node) {
	if len(nodes) <= 1 {
		return
	}

	// find node that has least number of children
	idx := leastChildren(nodes)
	if idx == -1 {
		return
	}
	tree := nodes[idx]
	treeLength := len(tree.Children)

	// allocate max able size for rightCommon slice
	// to get ability insert elements in reverse order (from end to start)
	// without sorting
	commonRight = make([]*ast.Node, treeLength)
	lastRight := treeLength // will use this to get results as commonRight[lastRight:]

	var (
		breakLeft   bool
		breakRight  bool
		commonTotal int
	)
	for i, j := 0, treeLength-1; commonTotal < treeLength && j >= 0 && !(breakLeft && breakRight); i, j = i+1, j-1 {
		treeLeft := tree.Children[i]
		treeRight := tree.Children[j]

		for k := 0; k < len(nodes) && !(breakLeft && breakRight); k++ {
			// skip least children node
			if k == idx {
				continue
			}

			restLeft := nodes[k].Children[i]
			restRight := nodes[k].Children[j+len(nodes[k].Children)-treeLength]

			breakLeft = breakLeft || !treeLeft.Equal(restLeft)

			// disable searching for right common parts, if left part is already overlapping
			breakRight = breakRight || (!breakLeft && j <= i)
			breakRight = breakRight || !treeRight.Equal(restRight)
		}

		if !breakLeft {
			commonTotal++
			commonLeft = append(commonLeft, treeLeft)
		}
		if !breakRight {
			commonTotal++
			lastRight = j
			commonRight[j] = treeRight
		}
	}

	commonRight = commonRight[lastRight:]

	return
}

func appendIfUnique(target []*ast.Node, val *ast.Node) []*ast.Node {
	for _, n := range target {
		if reflect.DeepEqual(n, val) {
			return target
		}
	}
	return append(target, val)
}

func areOfSameKind(nodes []*ast.Node, kind ast.Kind) bool {
	for _, n := range nodes {
		if n.Kind != kind {
			return false
		}
	}
	return true
}

func leastChildren(nodes []*ast.Node) int {
	min := -1
	idx := -1
	for i, n := range nodes {
		if idx == -1 || (len(n.Children) < min) {
			min = len(n.Children)
			idx = i
		}
	}
	return idx
}

func compileTreeChildren(tree *ast.Node, sep []rune) ([]match.Matcher, error) {
	var matchers []match.Matcher
	for _, desc := range tree.Children {
		m, err := compile(desc, sep)
		if err != nil {
			return nil, err
		}
		matchers = append(matchers, optimizeMatcher(m))
	}
	return matchers, nil
}

func compile(tree *ast.Node, sep []rune) (m match.Matcher, err error) {
	switch tree.Kind {
	case ast.KindAnyOf:
		// todo this could be faster on pattern_alternatives_combine_lite (see glob_test.go)
		if n := minimizeTree(tree); n != nil {
			return compile(n, sep)
		}
		matchers, err := compileTreeChildren(tree, sep)
		if err != nil {
			return nil, err
		}
		return match.NewAnyOf(matchers...), nil

	case ast.KindPattern:
		if len(tree.Children) == 0 {
			return match.NewNothing(), nil
		}
		matchers, err := compileTreeChildren(tree, sep)
		if err != nil {
			return nil, err
		}
		m, err = compileMatchers(minimizeMatchers(matchers))
		if err != nil {
			return nil, err
		}

	case ast.KindAny:
		m = match.NewAny(sep)

	case ast.KindSuper:
		m = match.NewSuper()

	case ast.KindSingle:
		m = match.NewSingle(sep)

	case ast.KindNothing:
		m = match.NewNothing()

	case ast.KindList:
		l := tree.Value.(ast.List)
		m = match.NewList([]rune(l.Chars), l.Not)

	case ast.KindRange:
		r := tree.Value.(ast.Range)
		m = match.NewRange(r.Lo, r.Hi, r.Not)

	case ast.KindText:
		t := tree.Value.(ast.Text)
		m = match.NewText(t.Text)

	default:
		return nil, fmt.Errorf("could not compile tree: unknown node type")
	}

	return optimizeMatcher(m), nil
}

func Compile(tree *ast.Node, sep []rune) (match.Matcher, error) {
	m, err := compile(tree, sep)
	if err != nil {
		return nil, err
	}

	return m, nil
}
