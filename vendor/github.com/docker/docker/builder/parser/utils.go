package parser

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// dumps the AST defined by `node` as a list of sexps. Returns a string
// suitable for printing.
func (node *Node) Dump() string {
	str := ""
	str += node.Value

	if len(node.Flags) > 0 {
		str += fmt.Sprintf(" %q", node.Flags)
	}

	for _, n := range node.Children {
		str += "(" + n.Dump() + ")\n"
	}

	if node.Next != nil {
		for n := node.Next; n != nil; n = n.Next {
			if len(n.Children) > 0 {
				str += " " + n.Dump()
			} else {
				str += " " + strconv.Quote(n.Value)
			}
		}
	}

	return strings.TrimSpace(str)
}

// performs the dispatch based on the two primal strings, cmd and args. Please
// look at the dispatch table in parser.go to see how these dispatchers work.
func fullDispatch(cmd, args string) (*Node, map[string]bool, error) {
	fn := dispatch[cmd]

	// Ignore invalid Dockerfile instructions
	if fn == nil {
		fn = parseIgnore
	}

	sexp, attrs, err := fn(args)
	if err != nil {
		return nil, nil, err
	}

	return sexp, attrs, nil
}

// splitCommand takes a single line of text and parses out the cmd and args,
// which are used for dispatching to more exact parsing functions.
func splitCommand(line string) (string, []string, string, error) {
	var args string
	var flags []string

	// Make sure we get the same results irrespective of leading/trailing spaces
	cmdline := TOKEN_WHITESPACE.Split(strings.TrimSpace(line), 2)
	cmd := strings.ToLower(cmdline[0])

	if len(cmdline) == 2 {
		var err error
		args, flags, err = extractBuilderFlags(cmdline[1])
		if err != nil {
			return "", nil, "", err
		}
	}

	return cmd, flags, strings.TrimSpace(args), nil
}

// covers comments and empty lines. Lines should be trimmed before passing to
// this function.
func stripComments(line string) string {
	// string is already trimmed at this point
	if TOKEN_COMMENT.MatchString(line) {
		return TOKEN_COMMENT.ReplaceAllString(line, "")
	}

	return line
}

func extractBuilderFlags(line string) (string, []string, error) {
	// Parses the BuilderFlags and returns the remaining part of the line

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

	for pos := 0; pos <= len(line); pos++ {
		if pos != len(line) {
			ch = rune(line[pos])
		}

		if phase == inSpaces { // Looking for start of word
			if pos == len(line) { // end of input
				break
			}
			if unicode.IsSpace(ch) { // skip spaces
				continue
			}

			// Only keep going if the next word starts with --
			if ch != '-' || pos+1 == len(line) || rune(line[pos+1]) != '-' {
				return line[pos:], words, nil
			}

			phase = inWord // found someting with "--", fall thru
		}
		if (phase == inWord || phase == inQuote) && (pos == len(line)) {
			if word != "--" && (blankOK || len(word) > 0) {
				words = append(words, word)
			}
			break
		}
		if phase == inWord {
			if unicode.IsSpace(ch) {
				phase = inSpaces
				if word == "--" {
					return line[pos:], words, nil
				}
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
				continue
			}
			if ch == '\\' {
				if pos+1 == len(line) {
					continue // just skip \ at end
				}
				pos++
				ch = rune(line[pos])
			}
			word += string(ch)
			continue
		}
		if phase == inQuote {
			if ch == quote {
				phase = inWord
				continue
			}
			if ch == '\\' {
				if pos+1 == len(line) {
					phase = inWord
					continue // just skip \ at end
				}
				pos++
				ch = rune(line[pos])
			}
			word += string(ch)
		}
	}

	return "", words, nil
}
