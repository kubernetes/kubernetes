// Copyright 2015 Light Code Labs, LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package caddyfile

import (
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// Parse parses the input just enough to group tokens, in
// order, by server block. No further parsing is performed.
// Server blocks are returned in the order in which they appear.
// Directives that do not appear in validDirectives will cause
// an error. If you do not want to check for valid directives,
// pass in nil instead.
func Parse(filename string, input io.Reader, validDirectives []string) ([]ServerBlock, error) {
	p := parser{Dispenser: NewDispenser(filename, input), validDirectives: validDirectives}
	return p.parseAll()
}

// allTokens lexes the entire input, but does not parse it.
// It returns all the tokens from the input, unstructured
// and in order.
func allTokens(input io.Reader) ([]Token, error) {
	l := new(lexer)
	err := l.load(input)
	if err != nil {
		return nil, err
	}
	var tokens []Token
	for l.next() {
		tokens = append(tokens, l.token)
	}
	return tokens, nil
}

type parser struct {
	Dispenser
	block           ServerBlock // current server block being parsed
	validDirectives []string    // a directive must be valid or it's an error
	eof             bool        // if we encounter a valid EOF in a hard place
	definedSnippets map[string][]Token
}

func (p *parser) parseAll() ([]ServerBlock, error) {
	var blocks []ServerBlock

	for p.Next() {
		err := p.parseOne()
		if err != nil {
			return blocks, err
		}
		if len(p.block.Keys) > 0 {
			blocks = append(blocks, p.block)
		}
	}

	return blocks, nil
}

func (p *parser) parseOne() error {
	p.block = ServerBlock{Tokens: make(map[string][]Token)}

	return p.begin()
}

func (p *parser) begin() error {
	if len(p.tokens) == 0 {
		return nil
	}

	err := p.addresses()

	if err != nil {
		return err
	}

	if p.eof {
		// this happens if the Caddyfile consists of only
		// a line of addresses and nothing else
		return nil
	}

	if ok, name := p.isSnippet(); ok {
		if p.definedSnippets == nil {
			p.definedSnippets = map[string][]Token{}
		}
		if _, found := p.definedSnippets[name]; found {
			return p.Errf("redeclaration of previously declared snippet %s", name)
		}
		// consume all tokens til matched close brace
		tokens, err := p.snippetTokens()
		if err != nil {
			return err
		}
		p.definedSnippets[name] = tokens
		// empty block keys so we don't save this block as a real server.
		p.block.Keys = nil
		return nil
	}

	return p.blockContents()
}

func (p *parser) addresses() error {
	var expectingAnother bool

	for {
		tkn := replaceEnvVars(p.Val())

		// special case: import directive replaces tokens during parse-time
		if tkn == "import" && p.isNewLine() {
			err := p.doImport()
			if err != nil {
				return err
			}
			continue
		}

		// Open brace definitely indicates end of addresses
		if tkn == "{" {
			if expectingAnother {
				return p.Errf("Expected another address but had '%s' - check for extra comma", tkn)
			}
			break
		}

		if tkn != "" { // empty token possible if user typed ""
			// Trailing comma indicates another address will follow, which
			// may possibly be on the next line
			if tkn[len(tkn)-1] == ',' {
				tkn = tkn[:len(tkn)-1]
				expectingAnother = true
			} else {
				expectingAnother = false // but we may still see another one on this line
			}

			p.block.Keys = append(p.block.Keys, tkn)
		}

		// Advance token and possibly break out of loop or return error
		hasNext := p.Next()
		if expectingAnother && !hasNext {
			return p.EOFErr()
		}
		if !hasNext {
			p.eof = true
			break // EOF
		}
		if !expectingAnother && p.isNewLine() {
			break
		}
	}

	return nil
}

func (p *parser) blockContents() error {
	errOpenCurlyBrace := p.openCurlyBrace()
	if errOpenCurlyBrace != nil {
		// single-server configs don't need curly braces
		p.cursor--
	}

	err := p.directives()
	if err != nil {
		return err
	}

	// Only look for close curly brace if there was an opening
	if errOpenCurlyBrace == nil {
		err = p.closeCurlyBrace()
		if err != nil {
			return err
		}
	}

	return nil
}

// directives parses through all the lines for directives
// and it expects the next token to be the first
// directive. It goes until EOF or closing curly brace
// which ends the server block.
func (p *parser) directives() error {
	for p.Next() {
		// end of server block
		if p.Val() == "}" {
			break
		}

		// special case: import directive replaces tokens during parse-time
		if p.Val() == "import" {
			err := p.doImport()
			if err != nil {
				return err
			}
			p.cursor-- // cursor is advanced when we continue, so roll back one more
			continue
		}

		// normal case: parse a directive on this line
		if err := p.directive(); err != nil {
			return err
		}
	}
	return nil
}

// doImport swaps out the import directive and its argument
// (a total of 2 tokens) with the tokens in the specified file
// or globbing pattern. When the function returns, the cursor
// is on the token before where the import directive was. In
// other words, call Next() to access the first token that was
// imported.
func (p *parser) doImport() error {
	// syntax checks
	if !p.NextArg() {
		return p.ArgErr()
	}
	importPattern := replaceEnvVars(p.Val())
	if importPattern == "" {
		return p.Err("Import requires a non-empty filepath")
	}
	if p.NextArg() {
		return p.Err("Import takes only one argument (glob pattern or file)")
	}
	// splice out the import directive and its argument (2 tokens total)
	tokensBefore := p.tokens[:p.cursor-1]
	tokensAfter := p.tokens[p.cursor+1:]
	var importedTokens []Token

	// first check snippets. That is a simple, non-recursive replacement
	if p.definedSnippets != nil && p.definedSnippets[importPattern] != nil {
		importedTokens = p.definedSnippets[importPattern]
	} else {
		// make path relative to the file of the _token_ being processed rather
		// than current working directory (issue #867) and then use glob to get
		// list of matching filenames
		absFile, err := filepath.Abs(p.Dispenser.File())
		if err != nil {
			return p.Errf("Failed to get absolute path of file: %s: %v", p.Dispenser.filename, err)
		}

		var matches []string
		var globPattern string
		if !filepath.IsAbs(importPattern) {
			globPattern = filepath.Join(filepath.Dir(absFile), importPattern)
		} else {
			globPattern = importPattern
		}
		if strings.Count(globPattern, "*") > 1 || strings.Count(globPattern, "?") > 1 ||
			(strings.Contains(globPattern, "[") && strings.Contains(globPattern, "]")) {
			// See issue #2096 - a pattern with many glob expansions can hang for too long
			return p.Errf("Glob pattern may only contain one wildcard (*), but has others: %s", globPattern)
		}
		matches, err = filepath.Glob(globPattern)

		if err != nil {
			return p.Errf("Failed to use import pattern %s: %v", importPattern, err)
		}
		if len(matches) == 0 {
			if strings.ContainsAny(globPattern, "*?[]") {
				log.Printf("[WARNING] No files matching import glob pattern: %s", importPattern)
			} else {
				return p.Errf("File to import not found: %s", importPattern)
			}
		}

		// collect all the imported tokens

		for _, importFile := range matches {
			newTokens, err := p.doSingleImport(importFile)
			if err != nil {
				return err
			}
			importedTokens = append(importedTokens, newTokens...)
		}
	}

	// splice the imported tokens in the place of the import statement
	// and rewind cursor so Next() will land on first imported token
	p.tokens = append(tokensBefore, append(importedTokens, tokensAfter...)...)
	p.cursor--

	return nil
}

// doSingleImport lexes the individual file at importFile and returns
// its tokens or an error, if any.
func (p *parser) doSingleImport(importFile string) ([]Token, error) {
	file, err := os.Open(importFile)
	if err != nil {
		return nil, p.Errf("Could not import %s: %v", importFile, err)
	}
	defer file.Close()

	if info, err := file.Stat(); err != nil {
		return nil, p.Errf("Could not import %s: %v", importFile, err)
	} else if info.IsDir() {
		return nil, p.Errf("Could not import %s: is a directory", importFile)
	}

	importedTokens, err := allTokens(file)
	if err != nil {
		return nil, p.Errf("Could not read tokens while importing %s: %v", importFile, err)
	}

	// Tack the file path onto these tokens so errors show the imported file's name
	// (we use full, absolute path to avoid bugs: issue #1892)
	filename, err := filepath.Abs(importFile)
	if err != nil {
		return nil, p.Errf("Failed to get absolute path of file: %s: %v", p.Dispenser.filename, err)
	}
	for i := 0; i < len(importedTokens); i++ {
		importedTokens[i].File = filename
	}

	return importedTokens, nil
}

// directive collects tokens until the directive's scope
// closes (either end of line or end of curly brace block).
// It expects the currently-loaded token to be a directive
// (or } that ends a server block). The collected tokens
// are loaded into the current server block for later use
// by directive setup functions.
func (p *parser) directive() error {
	dir := replaceEnvVars(p.Val())
	nesting := 0

	// TODO: More helpful error message ("did you mean..." or "maybe you need to install its server type")
	if !p.validDirective(dir) {
		return p.Errf("Unknown directive '%s'", dir)
	}

	// The directive itself is appended as a relevant token
	p.block.Tokens[dir] = append(p.block.Tokens[dir], p.tokens[p.cursor])

	for p.Next() {
		if p.Val() == "{" {
			nesting++
		} else if p.isNewLine() && nesting == 0 {
			p.cursor-- // read too far
			break
		} else if p.Val() == "}" && nesting > 0 {
			nesting--
		} else if p.Val() == "}" && nesting == 0 {
			return p.Err("Unexpected '}' because no matching opening brace")
		} else if p.Val() == "import" && p.isNewLine() {
			if err := p.doImport(); err != nil {
				return err
			}
			p.cursor-- // cursor is advanced when we continue, so roll back one more
			continue
		}
		p.tokens[p.cursor].Text = replaceEnvVars(p.tokens[p.cursor].Text)
		p.block.Tokens[dir] = append(p.block.Tokens[dir], p.tokens[p.cursor])
	}

	if nesting > 0 {
		return p.EOFErr()
	}
	return nil
}

// openCurlyBrace expects the current token to be an
// opening curly brace. This acts like an assertion
// because it returns an error if the token is not
// a opening curly brace. It does NOT advance the token.
func (p *parser) openCurlyBrace() error {
	if p.Val() != "{" {
		return p.SyntaxErr("{")
	}
	return nil
}

// closeCurlyBrace expects the current token to be
// a closing curly brace. This acts like an assertion
// because it returns an error if the token is not
// a closing curly brace. It does NOT advance the token.
func (p *parser) closeCurlyBrace() error {
	if p.Val() != "}" {
		return p.SyntaxErr("}")
	}
	return nil
}

// validDirective returns true if dir is in p.validDirectives.
func (p *parser) validDirective(dir string) bool {
	if p.validDirectives == nil {
		return true
	}
	for _, d := range p.validDirectives {
		if d == dir {
			return true
		}
	}
	return false
}

// replaceEnvVars replaces environment variables that appear in the token
// and understands both the $UNIX and %WINDOWS% syntaxes.
func replaceEnvVars(s string) string {
	s = replaceEnvReferences(s, "{%", "%}")
	s = replaceEnvReferences(s, "{$", "}")
	return s
}

// replaceEnvReferences performs the actual replacement of env variables
// in s, given the placeholder start and placeholder end strings.
func replaceEnvReferences(s, refStart, refEnd string) string {
	index := strings.Index(s, refStart)
	for index != -1 {
		endIndex := strings.Index(s[index:], refEnd)
		if endIndex == -1 {
			break
		}

		endIndex += index
		if endIndex > index+len(refStart) {
			ref := s[index : endIndex+len(refEnd)]
			s = strings.Replace(s, ref, os.Getenv(ref[len(refStart):len(ref)-len(refEnd)]), -1)
		} else {
			return s
		}
		index = strings.Index(s, refStart)
	}
	return s
}

// ServerBlock associates any number of keys (usually addresses
// of some sort) with tokens (grouped by directive name).
type ServerBlock struct {
	Keys   []string
	Tokens map[string][]Token
}

func (p *parser) isSnippet() (bool, string) {
	keys := p.block.Keys
	// A snippet block is a single key with parens. Nothing else qualifies.
	if len(keys) == 1 && strings.HasPrefix(keys[0], "(") && strings.HasSuffix(keys[0], ")") {
		return true, strings.TrimSuffix(keys[0][1:], ")")
	}
	return false, ""
}

// read and store everything in a block for later replay.
func (p *parser) snippetTokens() ([]Token, error) {
	// TODO: disallow imports in snippets for simplicity at import time
	// snippet must have curlies.
	err := p.openCurlyBrace()
	if err != nil {
		return nil, err
	}
	count := 1
	tokens := []Token{}
	for p.Next() {
		if p.Val() == "}" {
			count--
			if count == 0 {
				break
			}
		}
		if p.Val() == "{" {
			count++
		}
		tokens = append(tokens, p.tokens[p.cursor])
	}
	// make sure we're matched up
	if count != 0 {
		return nil, p.SyntaxErr("}")
	}
	return tokens, nil
}
