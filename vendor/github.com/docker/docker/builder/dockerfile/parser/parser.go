// Package parser implements a parser and parse tree dumper for Dockerfiles.
package parser

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"unicode"

	"github.com/docker/docker/builder/dockerfile/command"
	"github.com/docker/docker/pkg/system"
	"github.com/pkg/errors"
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
	StartLine  int             // the line in the original dockerfile where the node begins
	endLine    int             // the line in the original dockerfile where the node ends
}

// Dump dumps the AST defined by `node` as a list of sexps.
// Returns a string suitable for printing.
func (node *Node) Dump() string {
	str := ""
	str += node.Value

	if len(node.Flags) > 0 {
		str += fmt.Sprintf(" %q", node.Flags)
	}

	for _, n := range node.Children {
		str += "(" + n.Dump() + ")\n"
	}

	for n := node.Next; n != nil; n = n.Next {
		if len(n.Children) > 0 {
			str += " " + n.Dump()
		} else {
			str += " " + strconv.Quote(n.Value)
		}
	}

	return strings.TrimSpace(str)
}

func (node *Node) lines(start, end int) {
	node.StartLine = start
	node.endLine = end
}

// AddChild adds a new child node, and updates line information
func (node *Node) AddChild(child *Node, startLine, endLine int) {
	child.lines(startLine, endLine)
	if node.StartLine < 0 {
		node.StartLine = startLine
	}
	node.endLine = endLine
	node.Children = append(node.Children, child)
}

var (
	dispatch             map[string]func(string, *Directive) (*Node, map[string]bool, error)
	tokenWhitespace      = regexp.MustCompile(`[\t\v\f\r ]+`)
	tokenEscapeCommand   = regexp.MustCompile(`^#[ \t]*escape[ \t]*=[ \t]*(?P<escapechar>.).*$`)
	tokenPlatformCommand = regexp.MustCompile(`^#[ \t]*platform[ \t]*=[ \t]*(?P<platform>.*)$`)
	tokenComment         = regexp.MustCompile(`^#.*$`)
)

// DefaultEscapeToken is the default escape token
const DefaultEscapeToken = '\\'

// defaultPlatformToken is the platform assumed for the build if not explicitly provided
var defaultPlatformToken = runtime.GOOS

// Directive is the structure used during a build run to hold the state of
// parsing directives.
type Directive struct {
	escapeToken           rune           // Current escape token
	platformToken         string         // Current platform token
	lineContinuationRegex *regexp.Regexp // Current line continuation regex
	processingComplete    bool           // Whether we are done looking for directives
	escapeSeen            bool           // Whether the escape directive has been seen
	platformSeen          bool           // Whether the platform directive has been seen
}

// setEscapeToken sets the default token for escaping characters in a Dockerfile.
func (d *Directive) setEscapeToken(s string) error {
	if s != "`" && s != "\\" {
		return fmt.Errorf("invalid ESCAPE '%s'. Must be ` or \\", s)
	}
	d.escapeToken = rune(s[0])
	d.lineContinuationRegex = regexp.MustCompile(`\` + s + `[ \t]*$`)
	return nil
}

// setPlatformToken sets the default platform for pulling images in a Dockerfile.
func (d *Directive) setPlatformToken(s string) error {
	s = strings.ToLower(s)
	valid := []string{runtime.GOOS}
	if system.LCOWSupported() {
		valid = append(valid, "linux")
	}
	for _, item := range valid {
		if s == item {
			d.platformToken = s
			return nil
		}
	}
	return fmt.Errorf("invalid PLATFORM '%s'. Must be one of %v", s, valid)
}

// possibleParserDirective looks for one or more parser directives '# escapeToken=<char>' and
// '# platform=<string>'. Parser directives must precede any builder instruction
// or other comments, and cannot be repeated.
func (d *Directive) possibleParserDirective(line string) error {
	if d.processingComplete {
		return nil
	}

	tecMatch := tokenEscapeCommand.FindStringSubmatch(strings.ToLower(line))
	if len(tecMatch) != 0 {
		for i, n := range tokenEscapeCommand.SubexpNames() {
			if n == "escapechar" {
				if d.escapeSeen == true {
					return errors.New("only one escape parser directive can be used")
				}
				d.escapeSeen = true
				return d.setEscapeToken(tecMatch[i])
			}
		}
	}

	// TODO @jhowardmsft LCOW Support: Eventually this check can be removed,
	// but only recognise a platform token if running in LCOW mode.
	if system.LCOWSupported() {
		tpcMatch := tokenPlatformCommand.FindStringSubmatch(strings.ToLower(line))
		if len(tpcMatch) != 0 {
			for i, n := range tokenPlatformCommand.SubexpNames() {
				if n == "platform" {
					if d.platformSeen == true {
						return errors.New("only one platform parser directive can be used")
					}
					d.platformSeen = true
					return d.setPlatformToken(tpcMatch[i])
				}
			}
		}
	}

	d.processingComplete = true
	return nil
}

// NewDefaultDirective returns a new Directive with the default escapeToken token
func NewDefaultDirective() *Directive {
	directive := Directive{}
	directive.setEscapeToken(string(DefaultEscapeToken))
	directive.setPlatformToken(defaultPlatformToken)
	return &directive
}

func init() {
	// Dispatch Table. see line_parsers.go for the parse functions.
	// The command is parsed and mapped to the line parser. The line parser
	// receives the arguments but not the command, and returns an AST after
	// reformulating the arguments according to the rules in the parser
	// functions. Errors are propagated up by Parse() and the resulting AST can
	// be incorporated directly into the existing AST as a next.
	dispatch = map[string]func(string, *Directive) (*Node, map[string]bool, error){
		command.Add:         parseMaybeJSONToList,
		command.Arg:         parseNameOrNameVal,
		command.Cmd:         parseMaybeJSON,
		command.Copy:        parseMaybeJSONToList,
		command.Entrypoint:  parseMaybeJSON,
		command.Env:         parseEnv,
		command.Expose:      parseStringsWhitespaceDelimited,
		command.From:        parseStringsWhitespaceDelimited,
		command.Healthcheck: parseHealthConfig,
		command.Label:       parseLabel,
		command.Maintainer:  parseString,
		command.Onbuild:     parseSubCommand,
		command.Run:         parseMaybeJSON,
		command.Shell:       parseMaybeJSON,
		command.StopSignal:  parseString,
		command.User:        parseString,
		command.Volume:      parseMaybeJSONToList,
		command.Workdir:     parseString,
	}
}

// newNodeFromLine splits the line into parts, and dispatches to a function
// based on the command and command arguments. A Node is created from the
// result of the dispatch.
func newNodeFromLine(line string, directive *Directive) (*Node, error) {
	cmd, flags, args, err := splitCommand(line)
	if err != nil {
		return nil, err
	}

	fn := dispatch[cmd]
	// Ignore invalid Dockerfile instructions
	if fn == nil {
		fn = parseIgnore
	}
	next, attrs, err := fn(args, directive)
	if err != nil {
		return nil, err
	}

	return &Node{
		Value:      cmd,
		Original:   line,
		Flags:      flags,
		Next:       next,
		Attributes: attrs,
	}, nil
}

// Result is the result of parsing a Dockerfile
type Result struct {
	AST         *Node
	EscapeToken rune
	Platform    string
	Warnings    []string
}

// PrintWarnings to the writer
func (r *Result) PrintWarnings(out io.Writer) {
	if len(r.Warnings) == 0 {
		return
	}
	fmt.Fprintf(out, strings.Join(r.Warnings, "\n")+"\n")
}

// Parse reads lines from a Reader, parses the lines into an AST and returns
// the AST and escape token
func Parse(rwc io.Reader) (*Result, error) {
	d := NewDefaultDirective()
	currentLine := 0
	root := &Node{StartLine: -1}
	scanner := bufio.NewScanner(rwc)
	warnings := []string{}

	var err error
	for scanner.Scan() {
		bytesRead := scanner.Bytes()
		if currentLine == 0 {
			// First line, strip the byte-order-marker if present
			bytesRead = bytes.TrimPrefix(bytesRead, utf8bom)
		}
		bytesRead, err = processLine(d, bytesRead, true)
		if err != nil {
			return nil, err
		}
		currentLine++

		startLine := currentLine
		line, isEndOfLine := trimContinuationCharacter(string(bytesRead), d)
		if isEndOfLine && line == "" {
			continue
		}

		var hasEmptyContinuationLine bool
		for !isEndOfLine && scanner.Scan() {
			bytesRead, err := processLine(d, scanner.Bytes(), false)
			if err != nil {
				return nil, err
			}
			currentLine++

			if isEmptyContinuationLine(bytesRead) {
				hasEmptyContinuationLine = true
				continue
			}

			continuationLine := string(bytesRead)
			continuationLine, isEndOfLine = trimContinuationCharacter(continuationLine, d)
			line += continuationLine
		}

		if hasEmptyContinuationLine {
			warning := "[WARNING]: Empty continuation line found in:\n    " + line
			warnings = append(warnings, warning)
		}

		child, err := newNodeFromLine(line, d)
		if err != nil {
			return nil, err
		}
		root.AddChild(child, startLine, currentLine)
	}

	if len(warnings) > 0 {
		warnings = append(warnings, "[WARNING]: Empty continuation lines will become errors in a future release.")
	}
	return &Result{
		AST:         root,
		Warnings:    warnings,
		EscapeToken: d.escapeToken,
		Platform:    d.platformToken,
	}, nil
}

func trimComments(src []byte) []byte {
	return tokenComment.ReplaceAll(src, []byte{})
}

func trimWhitespace(src []byte) []byte {
	return bytes.TrimLeftFunc(src, unicode.IsSpace)
}

func isEmptyContinuationLine(line []byte) bool {
	return len(trimComments(trimWhitespace(line))) == 0
}

var utf8bom = []byte{0xEF, 0xBB, 0xBF}

func trimContinuationCharacter(line string, d *Directive) (string, bool) {
	if d.lineContinuationRegex.MatchString(line) {
		line = d.lineContinuationRegex.ReplaceAllString(line, "")
		return line, false
	}
	return line, true
}

// TODO: remove stripLeftWhitespace after deprecation period. It seems silly
// to preserve whitespace on continuation lines. Why is that done?
func processLine(d *Directive, token []byte, stripLeftWhitespace bool) ([]byte, error) {
	if stripLeftWhitespace {
		token = trimWhitespace(token)
	}
	return trimComments(token), d.possibleParserDirective(string(token))
}
