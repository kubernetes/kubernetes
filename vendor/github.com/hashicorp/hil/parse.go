package hil

import (
	"sync"

	"github.com/hashicorp/hil/ast"
)

var parserLock sync.Mutex
var parserResult ast.Node

// Parse parses the given program and returns an executable AST tree.
func Parse(v string) (ast.Node, error) {
	// Unfortunately due to the way that goyacc generated parsers are
	// formatted, we can only do a single parse at a time without a lot
	// of extra work. In the future we can remove this limitation.
	parserLock.Lock()
	defer parserLock.Unlock()

	// Reset our globals
	parserResult = nil

	// Create the lexer
	lex := &parserLex{Input: v}

	// Parse!
	parserParse(lex)

	return parserResult, lex.Err
}
