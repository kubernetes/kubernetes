package parser

// line parsers are dispatch calls that parse a single unit of text into a
// Node object which contains the whole statement. Dockerfiles have varied
// (but not usually unique, see ONBUILD for a unique example) parsing rules
// per-command, and these unify the processing in a way that makes it
// manageable.

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/docker/docker/builder/dockerfile/command"
)

var (
	errDockerfileNotStringArray = errors.New("When using JSON array syntax, arrays must be comprised of strings only.")
)

const (
	commandLabel = "LABEL"
)

// ignore the current argument. This will still leave a command parsed, but
// will not incorporate the arguments into the ast.
func parseIgnore(rest string, d *Directive) (*Node, map[string]bool, error) {
	return &Node{}, nil, nil
}

// used for onbuild. Could potentially be used for anything that represents a
// statement with sub-statements.
//
// ONBUILD RUN foo bar -> (onbuild (run foo bar))
//
func parseSubCommand(rest string, d *Directive) (*Node, map[string]bool, error) {
	if rest == "" {
		return nil, nil, nil
	}

	child, err := newNodeFromLine(rest, d)
	if err != nil {
		return nil, nil, err
	}

	return &Node{Children: []*Node{child}}, nil, nil
}

// helper to parse words (i.e space delimited or quoted strings) in a statement.
// The quotes are preserved as part of this function and they are stripped later
// as part of processWords().
func parseWords(rest string, d *Directive) []string {
	const (
		inSpaces = iota // looking for start of a word
		inWord
		inQuote
	)

	words := []string{}
	phase := inSpaces
	word := ""
	quote := '\000'
	blankOK := false
	var ch rune
	var chWidth int

	for pos := 0; pos <= len(rest); pos += chWidth {
		if pos != len(rest) {
			ch, chWidth = utf8.DecodeRuneInString(rest[pos:])
		}

		if phase == inSpaces { // Looking for start of word
			if pos == len(rest) { // end of input
				break
			}
			if unicode.IsSpace(ch) { // skip spaces
				continue
			}
			phase = inWord // found it, fall through
		}
		if (phase == inWord || phase == inQuote) && (pos == len(rest)) {
			if blankOK || len(word) > 0 {
				words = append(words, word)
			}
			break
		}
		if phase == inWord {
			if unicode.IsSpace(ch) {
				phase = inSpaces
				if blankOK || len(word) > 0 {
					words = append(words, word)
				}
				word = ""
				blankOK = false
				continue
			}
			if ch == '\'' || ch == '"' {
				quote = ch
				blankOK = true
				phase = inQuote
			}
			if ch == d.escapeToken {
				if pos+chWidth == len(rest) {
					continue // just skip an escape token at end of line
				}
				// If we're not quoted and we see an escape token, then always just
				// add the escape token plus the char to the word, even if the char
				// is a quote.
				word += string(ch)
				pos += chWidth
				ch, chWidth = utf8.DecodeRuneInString(rest[pos:])
			}
			word += string(ch)
			continue
		}
		if phase == inQuote {
			if ch == quote {
				phase = inWord
			}
			// The escape token is special except for ' quotes - can't escape anything for '
			if ch == d.escapeToken && quote != '\'' {
				if pos+chWidth == len(rest) {
					phase = inWord
					continue // just skip the escape token at end
				}
				pos += chWidth
				word += string(ch)
				ch, chWidth = utf8.DecodeRuneInString(rest[pos:])
			}
			word += string(ch)
		}
	}

	return words
}

// parse environment like statements. Note that this does *not* handle
// variable interpolation, which will be handled in the evaluator.
func parseNameVal(rest string, key string, d *Directive) (*Node, error) {
	// This is kind of tricky because we need to support the old
	// variant:   KEY name value
	// as well as the new one:    KEY name=value ...
	// The trigger to know which one is being used will be whether we hit
	// a space or = first.  space ==> old, "=" ==> new

	words := parseWords(rest, d)
	if len(words) == 0 {
		return nil, nil
	}

	// Old format (KEY name value)
	if !strings.Contains(words[0], "=") {
		parts := tokenWhitespace.Split(rest, 2)
		if len(parts) < 2 {
			return nil, fmt.Errorf(key + " must have two arguments")
		}
		return newKeyValueNode(parts[0], parts[1]), nil
	}

	var rootNode *Node
	var prevNode *Node
	for _, word := range words {
		if !strings.Contains(word, "=") {
			return nil, fmt.Errorf("Syntax error - can't find = in %q. Must be of the form: name=value", word)
		}

		parts := strings.SplitN(word, "=", 2)
		node := newKeyValueNode(parts[0], parts[1])
		rootNode, prevNode = appendKeyValueNode(node, rootNode, prevNode)
	}

	return rootNode, nil
}

func newKeyValueNode(key, value string) *Node {
	return &Node{
		Value: key,
		Next:  &Node{Value: value},
	}
}

func appendKeyValueNode(node, rootNode, prevNode *Node) (*Node, *Node) {
	if rootNode == nil {
		rootNode = node
	}
	if prevNode != nil {
		prevNode.Next = node
	}

	prevNode = node.Next
	return rootNode, prevNode
}

func parseEnv(rest string, d *Directive) (*Node, map[string]bool, error) {
	node, err := parseNameVal(rest, "ENV", d)
	return node, nil, err
}

func parseLabel(rest string, d *Directive) (*Node, map[string]bool, error) {
	node, err := parseNameVal(rest, commandLabel, d)
	return node, nil, err
}

// NodeFromLabels returns a Node for the injected labels
func NodeFromLabels(labels map[string]string) *Node {
	keys := []string{}
	for key := range labels {
		keys = append(keys, key)
	}
	// Sort the label to have a repeatable order
	sort.Strings(keys)

	labelPairs := []string{}
	var rootNode *Node
	var prevNode *Node
	for _, key := range keys {
		value := labels[key]
		labelPairs = append(labelPairs, fmt.Sprintf("%q='%s'", key, value))
		// Value must be single quoted to prevent env variable expansion
		// See https://github.com/docker/docker/issues/26027
		node := newKeyValueNode(key, "'"+value+"'")
		rootNode, prevNode = appendKeyValueNode(node, rootNode, prevNode)
	}

	return &Node{
		Value:    command.Label,
		Original: commandLabel + " " + strings.Join(labelPairs, " "),
		Next:     rootNode,
	}
}

// parses a statement containing one or more keyword definition(s) and/or
// value assignments, like `name1 name2= name3="" name4=value`.
// Note that this is a stricter format than the old format of assignment,
// allowed by parseNameVal(), in a way that this only allows assignment of the
// form `keyword=[<value>]` like  `name2=`, `name3=""`, and `name4=value` above.
// In addition, a keyword definition alone is of the form `keyword` like `name1`
// above. And the assignments `name2=` and `name3=""` are equivalent and
// assign an empty value to the respective keywords.
func parseNameOrNameVal(rest string, d *Directive) (*Node, map[string]bool, error) {
	words := parseWords(rest, d)
	if len(words) == 0 {
		return nil, nil, nil
	}

	var (
		rootnode *Node
		prevNode *Node
	)
	for i, word := range words {
		node := &Node{}
		node.Value = word
		if i == 0 {
			rootnode = node
		} else {
			prevNode.Next = node
		}
		prevNode = node
	}

	return rootnode, nil, nil
}

// parses a whitespace-delimited set of arguments. The result is effectively a
// linked list of string arguments.
func parseStringsWhitespaceDelimited(rest string, d *Directive) (*Node, map[string]bool, error) {
	if rest == "" {
		return nil, nil, nil
	}

	node := &Node{}
	rootnode := node
	prevnode := node
	for _, str := range tokenWhitespace.Split(rest, -1) { // use regexp
		prevnode = node
		node.Value = str
		node.Next = &Node{}
		node = node.Next
	}

	// XXX to get around regexp.Split *always* providing an empty string at the
	// end due to how our loop is constructed, nil out the last node in the
	// chain.
	prevnode.Next = nil

	return rootnode, nil, nil
}

// parseString just wraps the string in quotes and returns a working node.
func parseString(rest string, d *Directive) (*Node, map[string]bool, error) {
	if rest == "" {
		return nil, nil, nil
	}
	n := &Node{}
	n.Value = rest
	return n, nil, nil
}

// parseJSON converts JSON arrays to an AST.
func parseJSON(rest string, d *Directive) (*Node, map[string]bool, error) {
	rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	if !strings.HasPrefix(rest, "[") {
		return nil, nil, fmt.Errorf(`Error parsing "%s" as a JSON array`, rest)
	}

	var myJSON []interface{}
	if err := json.NewDecoder(strings.NewReader(rest)).Decode(&myJSON); err != nil {
		return nil, nil, err
	}

	var top, prev *Node
	for _, str := range myJSON {
		s, ok := str.(string)
		if !ok {
			return nil, nil, errDockerfileNotStringArray
		}

		node := &Node{Value: s}
		if prev == nil {
			top = node
		} else {
			prev.Next = node
		}
		prev = node
	}

	return top, map[string]bool{"json": true}, nil
}

// parseMaybeJSON determines if the argument appears to be a JSON array. If
// so, passes to parseJSON; if not, quotes the result and returns a single
// node.
func parseMaybeJSON(rest string, d *Directive) (*Node, map[string]bool, error) {
	if rest == "" {
		return nil, nil, nil
	}

	node, attrs, err := parseJSON(rest, d)

	if err == nil {
		return node, attrs, nil
	}
	if err == errDockerfileNotStringArray {
		return nil, nil, err
	}

	node = &Node{}
	node.Value = rest
	return node, nil, nil
}

// parseMaybeJSONToList determines if the argument appears to be a JSON array. If
// so, passes to parseJSON; if not, attempts to parse it as a whitespace
// delimited string.
func parseMaybeJSONToList(rest string, d *Directive) (*Node, map[string]bool, error) {
	node, attrs, err := parseJSON(rest, d)

	if err == nil {
		return node, attrs, nil
	}
	if err == errDockerfileNotStringArray {
		return nil, nil, err
	}

	return parseStringsWhitespaceDelimited(rest, d)
}

// The HEALTHCHECK command is like parseMaybeJSON, but has an extra type argument.
func parseHealthConfig(rest string, d *Directive) (*Node, map[string]bool, error) {
	// Find end of first argument
	var sep int
	for ; sep < len(rest); sep++ {
		if unicode.IsSpace(rune(rest[sep])) {
			break
		}
	}
	next := sep
	for ; next < len(rest); next++ {
		if !unicode.IsSpace(rune(rest[next])) {
			break
		}
	}

	if sep == 0 {
		return nil, nil, nil
	}

	typ := rest[:sep]
	cmd, attrs, err := parseMaybeJSON(rest[next:], d)
	if err != nil {
		return nil, nil, err
	}

	return &Node{Value: typ, Next: cmd}, attrs, err
}
