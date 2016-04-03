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
	"strings"
	"unicode"
)

var (
	errDockerfileNotStringArray = errors.New("When using JSON array syntax, arrays must be comprised of strings only.")
)

// ignore the current argument. This will still leave a command parsed, but
// will not incorporate the arguments into the ast.
func parseIgnore(rest string) (*Node, map[string]bool, error) {
	return &Node{}, nil, nil
}

// used for onbuild. Could potentially be used for anything that represents a
// statement with sub-statements.
//
// ONBUILD RUN foo bar -> (onbuild (run foo bar))
//
func parseSubCommand(rest string) (*Node, map[string]bool, error) {
	if rest == "" {
		return nil, nil, nil
	}

	_, child, err := parseLine(rest)
	if err != nil {
		return nil, nil, err
	}

	return &Node{Children: []*Node{child}}, nil, nil
}

// parse environment like statements. Note that this does *not* handle
// variable interpolation, which will be handled in the evaluator.
func parseNameVal(rest string, key string) (*Node, map[string]bool, error) {
	// This is kind of tricky because we need to support the old
	// variant:   KEY name value
	// as well as the new one:    KEY name=value ...
	// The trigger to know which one is being used will be whether we hit
	// a space or = first.  space ==> old, "=" ==> new

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

	for pos := 0; pos <= len(rest); pos++ {
		if pos != len(rest) {
			ch = rune(rest[pos])
		}

		if phase == inSpaces { // Looking for start of word
			if pos == len(rest) { // end of input
				break
			}
			if unicode.IsSpace(ch) { // skip spaces
				continue
			}
			phase = inWord // found it, fall thru
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

					// Look for = and if not there assume
					// we're doing the old stuff and
					// just read the rest of the line
					if !strings.Contains(word, "=") {
						word = strings.TrimSpace(rest[pos:])
						words = append(words, word)
						break
					}
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
			if ch == '\\' {
				if pos+1 == len(rest) {
					continue // just skip \ at end
				}
				// If we're not quoted and we see a \, then always just
				// add \ plus the char to the word, even if the char
				// is a quote.
				word += string(ch)
				pos++
				ch = rune(rest[pos])
			}
			word += string(ch)
			continue
		}
		if phase == inQuote {
			if ch == quote {
				phase = inWord
			}
			// \ is special except for ' quotes - can't escape anything for '
			if ch == '\\' && quote != '\'' {
				if pos+1 == len(rest) {
					phase = inWord
					continue // just skip \ at end
				}
				pos++
				nextCh := rune(rest[pos])
				word += string(ch)
				ch = nextCh
			}
			word += string(ch)
		}
	}

	if len(words) == 0 {
		return nil, nil, nil
	}

	// Old format (KEY name value)
	var rootnode *Node

	if !strings.Contains(words[0], "=") {
		node := &Node{}
		rootnode = node
		strs := TOKEN_WHITESPACE.Split(rest, 2)

		if len(strs) < 2 {
			return nil, nil, fmt.Errorf(key + " must have two arguments")
		}

		node.Value = strs[0]
		node.Next = &Node{}
		node.Next.Value = strs[1]
	} else {
		var prevNode *Node
		for i, word := range words {
			if !strings.Contains(word, "=") {
				return nil, nil, fmt.Errorf("Syntax error - can't find = in %q. Must be of the form: name=value", word)
			}
			parts := strings.SplitN(word, "=", 2)

			name := &Node{}
			value := &Node{}

			name.Next = value
			name.Value = parts[0]
			value.Value = parts[1]

			if i == 0 {
				rootnode = name
			} else {
				prevNode.Next = name
			}
			prevNode = value
		}
	}

	return rootnode, nil, nil
}

func parseEnv(rest string) (*Node, map[string]bool, error) {
	return parseNameVal(rest, "ENV")
}

func parseLabel(rest string) (*Node, map[string]bool, error) {
	return parseNameVal(rest, "LABEL")
}

// parses a whitespace-delimited set of arguments. The result is effectively a
// linked list of string arguments.
func parseStringsWhitespaceDelimited(rest string) (*Node, map[string]bool, error) {
	if rest == "" {
		return nil, nil, nil
	}

	node := &Node{}
	rootnode := node
	prevnode := node
	for _, str := range TOKEN_WHITESPACE.Split(rest, -1) { // use regexp
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

// parsestring just wraps the string in quotes and returns a working node.
func parseString(rest string) (*Node, map[string]bool, error) {
	if rest == "" {
		return nil, nil, nil
	}
	n := &Node{}
	n.Value = rest
	return n, nil, nil
}

// parseJSON converts JSON arrays to an AST.
func parseJSON(rest string) (*Node, map[string]bool, error) {
	var myJson []interface{}
	if err := json.NewDecoder(strings.NewReader(rest)).Decode(&myJson); err != nil {
		return nil, nil, err
	}

	var top, prev *Node
	for _, str := range myJson {
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
func parseMaybeJSON(rest string) (*Node, map[string]bool, error) {
	if rest == "" {
		return nil, nil, nil
	}

	node, attrs, err := parseJSON(rest)

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
func parseMaybeJSONToList(rest string) (*Node, map[string]bool, error) {
	node, attrs, err := parseJSON(rest)

	if err == nil {
		return node, attrs, nil
	}
	if err == errDockerfileNotStringArray {
		return nil, nil, err
	}

	return parseStringsWhitespaceDelimited(rest)
}
