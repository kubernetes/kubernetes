// This package implements a parser and parse tree dumper for Dockerfiles.
package parser

import (
	"bufio"
	"io"
	"regexp"
	"strings"
	"unicode"

	"github.com/docker/docker/builder/command"
)

// Node is a structure used to represent a parse tree.
//
// In the node there are three fields, Value, Next, and Children. Value is the
// current token's string value. Next is always the next non-child token, and
// children contains all the children. Here's an example:
//
// (value next (child child-next child-next-next) next-next)
//
// This data structure is frankly pretty lousy for handling complex languages,
// but lucky for us the Dockerfile isn't very complicated. This structure
// works a little more effectively than a "proper" parse tree for our needs.
//
type Node struct {
	Value      string          // actual content
	Next       *Node           // the next item in the current sexp
	Children   []*Node         // the children of this sexp
	Attributes map[string]bool // special attributes for this node
	Original   string          // original line used before parsing
	Flags      []string        // only top Node should have this set
}

var (
	dispatch                map[string]func(string) (*Node, map[string]bool, error)
	TOKEN_WHITESPACE        = regexp.MustCompile(`[\t\v\f\r ]+`)
	TOKEN_LINE_CONTINUATION = regexp.MustCompile(`\\[ \t]*$`)
	TOKEN_COMMENT           = regexp.MustCompile(`^#.*$`)
)

func init() {
	// Dispatch Table. see line_parsers.go for the parse functions.
	// The command is parsed and mapped to the line parser. The line parser
	// recieves the arguments but not the command, and returns an AST after
	// reformulating the arguments according to the rules in the parser
	// functions. Errors are propagated up by Parse() and the resulting AST can
	// be incorporated directly into the existing AST as a next.
	dispatch = map[string]func(string) (*Node, map[string]bool, error){
		command.User:       parseString,
		command.Onbuild:    parseSubCommand,
		command.Workdir:    parseString,
		command.Env:        parseEnv,
		command.Label:      parseLabel,
		command.Maintainer: parseString,
		command.From:       parseString,
		command.Add:        parseMaybeJSONToList,
		command.Copy:       parseMaybeJSONToList,
		command.Run:        parseMaybeJSON,
		command.Cmd:        parseMaybeJSON,
		command.Entrypoint: parseMaybeJSON,
		command.Expose:     parseStringsWhitespaceDelimited,
		command.Volume:     parseMaybeJSONToList,
	}
}

// parse a line and return the remainder.
func parseLine(line string) (string, *Node, error) {
	if line = stripComments(line); line == "" {
		return "", nil, nil
	}

	if TOKEN_LINE_CONTINUATION.MatchString(line) {
		line = TOKEN_LINE_CONTINUATION.ReplaceAllString(line, "")
		return line, nil, nil
	}

	cmd, flags, args, err := splitCommand(line)
	if err != nil {
		return "", nil, err
	}

	node := &Node{}
	node.Value = cmd

	sexp, attrs, err := fullDispatch(cmd, args)
	if err != nil {
		return "", nil, err
	}

	node.Next = sexp
	node.Attributes = attrs
	node.Original = line
	node.Flags = flags

	return "", node, nil
}

// The main parse routine. Handles an io.ReadWriteCloser and returns the root
// of the AST.
func Parse(rwc io.Reader) (*Node, error) {
	root := &Node{}
	scanner := bufio.NewScanner(rwc)

	for scanner.Scan() {
		scannedLine := strings.TrimLeftFunc(scanner.Text(), unicode.IsSpace)
		line, child, err := parseLine(scannedLine)
		if err != nil {
			return nil, err
		}

		if line != "" && child == nil {
			for scanner.Scan() {
				newline := scanner.Text()

				if stripComments(strings.TrimSpace(newline)) == "" {
					continue
				}

				line, child, err = parseLine(line + newline)
				if err != nil {
					return nil, err
				}

				if child != nil {
					break
				}
			}
			if child == nil && line != "" {
				line, child, err = parseLine(line)
				if err != nil {
					return nil, err
				}
			}
		}

		if child != nil {
			root.Children = append(root.Children, child)
		}
	}

	return root, nil
}
