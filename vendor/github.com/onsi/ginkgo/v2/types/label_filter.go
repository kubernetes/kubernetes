package types

import (
	"fmt"
	"regexp"
	"strings"
)

var DEBUG_LABEL_FILTER_PARSING = false

type LabelFilter func([]string) bool

func matchLabelAction(label string) LabelFilter {
	expected := strings.ToLower(label)
	return func(labels []string) bool {
		for i := range labels {
			if strings.ToLower(labels[i]) == expected {
				return true
			}
		}
		return false
	}
}

func matchLabelRegexAction(regex *regexp.Regexp) LabelFilter {
	return func(labels []string) bool {
		for i := range labels {
			if regex.MatchString(labels[i]) {
				return true
			}
		}
		return false
	}
}

func notAction(filter LabelFilter) LabelFilter {
	return func(labels []string) bool { return !filter(labels) }
}

func andAction(a, b LabelFilter) LabelFilter {
	return func(labels []string) bool { return a(labels) && b(labels) }
}

func orAction(a, b LabelFilter) LabelFilter {
	return func(labels []string) bool { return a(labels) || b(labels) }
}

type lfToken uint

const (
	lfTokenInvalid lfToken = iota

	lfTokenRoot
	lfTokenOpenGroup
	lfTokenCloseGroup
	lfTokenNot
	lfTokenAnd
	lfTokenOr
	lfTokenRegexp
	lfTokenLabel
	lfTokenEOF
)

func (l lfToken) Precedence() int {
	switch l {
	case lfTokenRoot, lfTokenOpenGroup:
		return 0
	case lfTokenOr:
		return 1
	case lfTokenAnd:
		return 2
	case lfTokenNot:
		return 3
	}
	return -1
}

func (l lfToken) String() string {
	switch l {
	case lfTokenRoot:
		return "ROOT"
	case lfTokenOpenGroup:
		return "("
	case lfTokenCloseGroup:
		return ")"
	case lfTokenNot:
		return "!"
	case lfTokenAnd:
		return "&&"
	case lfTokenOr:
		return "||"
	case lfTokenRegexp:
		return "/regexp/"
	case lfTokenLabel:
		return "label"
	case lfTokenEOF:
		return "EOF"
	}
	return "INVALID"
}

type treeNode struct {
	token    lfToken
	location int
	value    string

	parent    *treeNode
	leftNode  *treeNode
	rightNode *treeNode
}

func (tn *treeNode) setRightNode(node *treeNode) {
	tn.rightNode = node
	node.parent = tn
}

func (tn *treeNode) setLeftNode(node *treeNode) {
	tn.leftNode = node
	node.parent = tn
}

func (tn *treeNode) firstAncestorWithPrecedenceLEQ(precedence int) *treeNode {
	if tn.token.Precedence() <= precedence {
		return tn
	}
	return tn.parent.firstAncestorWithPrecedenceLEQ(precedence)
}

func (tn *treeNode) firstUnmatchedOpenNode() *treeNode {
	if tn.token == lfTokenOpenGroup {
		return tn
	}
	if tn.parent == nil {
		return nil
	}
	return tn.parent.firstUnmatchedOpenNode()
}

func (tn *treeNode) constructLabelFilter(input string) (LabelFilter, error) {
	switch tn.token {
	case lfTokenOpenGroup:
		return nil, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, tn.location, "Mismatched '(' - could not find matching ')'.")
	case lfTokenLabel:
		return matchLabelAction(tn.value), nil
	case lfTokenRegexp:
		re, err := regexp.Compile(tn.value)
		if err != nil {
			return nil, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, tn.location, fmt.Sprintf("RegExp compilation error: %s", err))
		}
		return matchLabelRegexAction(re), nil
	}

	if tn.rightNode == nil {
		return nil, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, -1, "Unexpected EOF.")
	}
	rightLF, err := tn.rightNode.constructLabelFilter(input)
	if err != nil {
		return nil, err
	}

	switch tn.token {
	case lfTokenRoot, lfTokenCloseGroup:
		return rightLF, nil
	case lfTokenNot:
		return notAction(rightLF), nil
	}

	if tn.leftNode == nil {
		return nil, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, tn.location, fmt.Sprintf("Malformed tree - '%s' is missing left operand.", tn.token))
	}
	leftLF, err := tn.leftNode.constructLabelFilter(input)
	if err != nil {
		return nil, err
	}

	switch tn.token {
	case lfTokenAnd:
		return andAction(leftLF, rightLF), nil
	case lfTokenOr:
		return orAction(leftLF, rightLF), nil
	}

	return nil, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, tn.location, fmt.Sprintf("Invalid token '%s'.", tn.token))
}

func (tn *treeNode) tokenString() string {
	out := fmt.Sprintf("<%s", tn.token)
	if tn.value != "" {
		out += " | " + tn.value
	}
	out += ">"
	return out
}

func (tn *treeNode) toString(indent int) string {
	out := tn.tokenString() + "\n"
	if tn.leftNode != nil {
		out += fmt.Sprintf("%s  |_(L)_%s", strings.Repeat(" ", indent), tn.leftNode.toString(indent+1))
	}
	if tn.rightNode != nil {
		out += fmt.Sprintf("%s  |_(R)_%s", strings.Repeat(" ", indent), tn.rightNode.toString(indent+1))
	}
	return out
}

func tokenize(input string) func() (*treeNode, error) {
	runes, i := []rune(input), 0

	peekIs := func(r rune) bool {
		if i+1 < len(runes) {
			return runes[i+1] == r
		}
		return false
	}

	consumeUntil := func(cutset string) (string, int) {
		j := i
		for ; j < len(runes); j++ {
			if strings.IndexRune(cutset, runes[j]) >= 0 {
				break
			}
		}
		return string(runes[i:j]), j - i
	}

	return func() (*treeNode, error) {
		for i < len(runes) && runes[i] == ' ' {
			i += 1
		}

		if i >= len(runes) {
			return &treeNode{token: lfTokenEOF}, nil
		}

		node := &treeNode{location: i}
		switch runes[i] {
		case '&':
			if !peekIs('&') {
				return &treeNode{}, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, i, "Invalid token '&'.  Did you mean '&&'?")
			}
			i += 2
			node.token = lfTokenAnd
		case '|':
			if !peekIs('|') {
				return &treeNode{}, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, i, "Invalid token '|'.  Did you mean '||'?")
			}
			i += 2
			node.token = lfTokenOr
		case '!':
			i += 1
			node.token = lfTokenNot
		case ',':
			i += 1
			node.token = lfTokenOr
		case '(':
			i += 1
			node.token = lfTokenOpenGroup
		case ')':
			i += 1
			node.token = lfTokenCloseGroup
		case '/':
			i += 1
			value, n := consumeUntil("/")
			i += n + 1
			node.token, node.value = lfTokenRegexp, value
		default:
			value, n := consumeUntil("&|!,()/")
			i += n
			node.token, node.value = lfTokenLabel, strings.TrimSpace(value)
		}
		return node, nil
	}
}

func MustParseLabelFilter(input string) LabelFilter {
	filter, err := ParseLabelFilter(input)
	if err != nil {
		panic(err)
	}
	return filter
}

func ParseLabelFilter(input string) (LabelFilter, error) {
	if DEBUG_LABEL_FILTER_PARSING {
		fmt.Println("\n==============")
		fmt.Println("Input: ", input)
		fmt.Print("Tokens: ")
	}
	if input == "" {
		return func(_ []string) bool { return true }, nil
	}
	nextToken := tokenize(input)

	root := &treeNode{token: lfTokenRoot}
	current := root
LOOP:
	for {
		node, err := nextToken()
		if err != nil {
			return nil, err
		}

		if DEBUG_LABEL_FILTER_PARSING {
			fmt.Print(node.tokenString() + " ")
		}

		switch node.token {
		case lfTokenEOF:
			break LOOP
		case lfTokenLabel, lfTokenRegexp:
			if current.rightNode != nil {
				return nil, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, node.location, "Found two adjacent labels.  You need an operator between them.")
			}
			current.setRightNode(node)
		case lfTokenNot, lfTokenOpenGroup:
			if current.rightNode != nil {
				return nil, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, node.location, fmt.Sprintf("Invalid token '%s'.", node.token))
			}
			current.setRightNode(node)
			current = node
		case lfTokenAnd, lfTokenOr:
			if current.rightNode == nil {
				return nil, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, node.location, fmt.Sprintf("Operator '%s' missing left hand operand.", node.token))
			}
			nodeToStealFrom := current.firstAncestorWithPrecedenceLEQ(node.token.Precedence())
			node.setLeftNode(nodeToStealFrom.rightNode)
			nodeToStealFrom.setRightNode(node)
			current = node
		case lfTokenCloseGroup:
			firstUnmatchedOpenNode := current.firstUnmatchedOpenNode()
			if firstUnmatchedOpenNode == nil {
				return nil, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, node.location, "Mismatched ')' - could not find matching '('.")
			}
			if firstUnmatchedOpenNode == current && current.rightNode == nil {
				return nil, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, node.location, "Found empty '()' group.")
			}
			firstUnmatchedOpenNode.token = lfTokenCloseGroup //signify the group is now closed
			current = firstUnmatchedOpenNode.parent
		default:
			return nil, GinkgoErrors.SyntaxErrorParsingLabelFilter(input, node.location, fmt.Sprintf("Unknown token '%s'.", node.token))
		}
	}
	if DEBUG_LABEL_FILTER_PARSING {
		fmt.Printf("\n Tree:\n%s", root.toString(0))
	}
	return root.constructLabelFilter(input)
}

func ValidateAndCleanupLabel(label string, cl CodeLocation) (string, error) {
	out := strings.TrimSpace(label)
	if out == "" {
		return "", GinkgoErrors.InvalidEmptyLabel(cl)
	}
	if strings.ContainsAny(out, "&|!,()/") {
		return "", GinkgoErrors.InvalidLabel(label, cl)
	}
	return out, nil
}
