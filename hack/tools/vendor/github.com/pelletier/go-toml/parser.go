// TOML Parser.

package toml

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"time"
)

type tomlParser struct {
	flowIdx       int
	flow          []token
	tree          *Tree
	currentTable  []string
	seenTableKeys []string
}

type tomlParserStateFn func() tomlParserStateFn

// Formats and panics an error message based on a token
func (p *tomlParser) raiseError(tok *token, msg string, args ...interface{}) {
	panic(tok.Position.String() + ": " + fmt.Sprintf(msg, args...))
}

func (p *tomlParser) run() {
	for state := p.parseStart; state != nil; {
		state = state()
	}
}

func (p *tomlParser) peek() *token {
	if p.flowIdx >= len(p.flow) {
		return nil
	}
	return &p.flow[p.flowIdx]
}

func (p *tomlParser) assume(typ tokenType) {
	tok := p.getToken()
	if tok == nil {
		p.raiseError(tok, "was expecting token %s, but token stream is empty", tok)
	}
	if tok.typ != typ {
		p.raiseError(tok, "was expecting token %s, but got %s instead", typ, tok)
	}
}

func (p *tomlParser) getToken() *token {
	tok := p.peek()
	if tok == nil {
		return nil
	}
	p.flowIdx++
	return tok
}

func (p *tomlParser) parseStart() tomlParserStateFn {
	tok := p.peek()

	// end of stream, parsing is finished
	if tok == nil {
		return nil
	}

	switch tok.typ {
	case tokenDoubleLeftBracket:
		return p.parseGroupArray
	case tokenLeftBracket:
		return p.parseGroup
	case tokenKey:
		return p.parseAssign
	case tokenEOF:
		return nil
	case tokenError:
		p.raiseError(tok, "parsing error: %s", tok.String())
	default:
		p.raiseError(tok, "unexpected token %s", tok.typ)
	}
	return nil
}

func (p *tomlParser) parseGroupArray() tomlParserStateFn {
	startToken := p.getToken() // discard the [[
	key := p.getToken()
	if key.typ != tokenKeyGroupArray {
		p.raiseError(key, "unexpected token %s, was expecting a table array key", key)
	}

	// get or create table array element at the indicated part in the path
	keys, err := parseKey(key.val)
	if err != nil {
		p.raiseError(key, "invalid table array key: %s", err)
	}
	p.tree.createSubTree(keys[:len(keys)-1], startToken.Position) // create parent entries
	destTree := p.tree.GetPath(keys)
	var array []*Tree
	if destTree == nil {
		array = make([]*Tree, 0)
	} else if target, ok := destTree.([]*Tree); ok && target != nil {
		array = destTree.([]*Tree)
	} else {
		p.raiseError(key, "key %s is already assigned and not of type table array", key)
	}
	p.currentTable = keys

	// add a new tree to the end of the table array
	newTree := newTree()
	newTree.position = startToken.Position
	array = append(array, newTree)
	p.tree.SetPath(p.currentTable, array)

	// remove all keys that were children of this table array
	prefix := key.val + "."
	found := false
	for ii := 0; ii < len(p.seenTableKeys); {
		tableKey := p.seenTableKeys[ii]
		if strings.HasPrefix(tableKey, prefix) {
			p.seenTableKeys = append(p.seenTableKeys[:ii], p.seenTableKeys[ii+1:]...)
		} else {
			found = (tableKey == key.val)
			ii++
		}
	}

	// keep this key name from use by other kinds of assignments
	if !found {
		p.seenTableKeys = append(p.seenTableKeys, key.val)
	}

	// move to next parser state
	p.assume(tokenDoubleRightBracket)
	return p.parseStart
}

func (p *tomlParser) parseGroup() tomlParserStateFn {
	startToken := p.getToken() // discard the [
	key := p.getToken()
	if key.typ != tokenKeyGroup {
		p.raiseError(key, "unexpected token %s, was expecting a table key", key)
	}
	for _, item := range p.seenTableKeys {
		if item == key.val {
			p.raiseError(key, "duplicated tables")
		}
	}

	p.seenTableKeys = append(p.seenTableKeys, key.val)
	keys, err := parseKey(key.val)
	if err != nil {
		p.raiseError(key, "invalid table array key: %s", err)
	}
	if err := p.tree.createSubTree(keys, startToken.Position); err != nil {
		p.raiseError(key, "%s", err)
	}
	destTree := p.tree.GetPath(keys)
	if target, ok := destTree.(*Tree); ok && target != nil && target.inline {
		p.raiseError(key, "could not re-define exist inline table or its sub-table : %s",
			strings.Join(keys, "."))
	}
	p.assume(tokenRightBracket)
	p.currentTable = keys
	return p.parseStart
}

func (p *tomlParser) parseAssign() tomlParserStateFn {
	key := p.getToken()
	p.assume(tokenEqual)

	parsedKey, err := parseKey(key.val)
	if err != nil {
		p.raiseError(key, "invalid key: %s", err.Error())
	}

	value := p.parseRvalue()
	var tableKey []string
	if len(p.currentTable) > 0 {
		tableKey = p.currentTable
	} else {
		tableKey = []string{}
	}

	prefixKey := parsedKey[0 : len(parsedKey)-1]
	tableKey = append(tableKey, prefixKey...)

	// find the table to assign, looking out for arrays of tables
	var targetNode *Tree
	switch node := p.tree.GetPath(tableKey).(type) {
	case []*Tree:
		targetNode = node[len(node)-1]
	case *Tree:
		targetNode = node
	case nil:
		// create intermediate
		if err := p.tree.createSubTree(tableKey, key.Position); err != nil {
			p.raiseError(key, "could not create intermediate group: %s", err)
		}
		targetNode = p.tree.GetPath(tableKey).(*Tree)
	default:
		p.raiseError(key, "Unknown table type for path: %s",
			strings.Join(tableKey, "."))
	}

	if targetNode.inline {
		p.raiseError(key, "could not add key or sub-table to exist inline table or its sub-table : %s",
			strings.Join(tableKey, "."))
	}

	// assign value to the found table
	keyVal := parsedKey[len(parsedKey)-1]
	localKey := []string{keyVal}
	finalKey := append(tableKey, keyVal)
	if targetNode.GetPath(localKey) != nil {
		p.raiseError(key, "The following key was defined twice: %s",
			strings.Join(finalKey, "."))
	}
	var toInsert interface{}

	switch value.(type) {
	case *Tree, []*Tree:
		toInsert = value
	default:
		toInsert = &tomlValue{value: value, position: key.Position}
	}
	targetNode.values[keyVal] = toInsert
	return p.parseStart
}

var errInvalidUnderscore = errors.New("invalid use of _ in number")

func numberContainsInvalidUnderscore(value string) error {
	// For large numbers, you may use underscores between digits to enhance
	// readability. Each underscore must be surrounded by at least one digit on
	// each side.

	hasBefore := false
	for idx, r := range value {
		if r == '_' {
			if !hasBefore || idx+1 >= len(value) {
				// can't end with an underscore
				return errInvalidUnderscore
			}
		}
		hasBefore = isDigit(r)
	}
	return nil
}

var errInvalidUnderscoreHex = errors.New("invalid use of _ in hex number")

func hexNumberContainsInvalidUnderscore(value string) error {
	hasBefore := false
	for idx, r := range value {
		if r == '_' {
			if !hasBefore || idx+1 >= len(value) {
				// can't end with an underscore
				return errInvalidUnderscoreHex
			}
		}
		hasBefore = isHexDigit(r)
	}
	return nil
}

func cleanupNumberToken(value string) string {
	cleanedVal := strings.Replace(value, "_", "", -1)
	return cleanedVal
}

func (p *tomlParser) parseRvalue() interface{} {
	tok := p.getToken()
	if tok == nil || tok.typ == tokenEOF {
		p.raiseError(tok, "expecting a value")
	}

	switch tok.typ {
	case tokenString:
		return tok.val
	case tokenTrue:
		return true
	case tokenFalse:
		return false
	case tokenInf:
		if tok.val[0] == '-' {
			return math.Inf(-1)
		}
		return math.Inf(1)
	case tokenNan:
		return math.NaN()
	case tokenInteger:
		cleanedVal := cleanupNumberToken(tok.val)
		var err error
		var val int64
		if len(cleanedVal) >= 3 && cleanedVal[0] == '0' {
			switch cleanedVal[1] {
			case 'x':
				err = hexNumberContainsInvalidUnderscore(tok.val)
				if err != nil {
					p.raiseError(tok, "%s", err)
				}
				val, err = strconv.ParseInt(cleanedVal[2:], 16, 64)
			case 'o':
				err = numberContainsInvalidUnderscore(tok.val)
				if err != nil {
					p.raiseError(tok, "%s", err)
				}
				val, err = strconv.ParseInt(cleanedVal[2:], 8, 64)
			case 'b':
				err = numberContainsInvalidUnderscore(tok.val)
				if err != nil {
					p.raiseError(tok, "%s", err)
				}
				val, err = strconv.ParseInt(cleanedVal[2:], 2, 64)
			default:
				panic("invalid base") // the lexer should catch this first
			}
		} else {
			err = numberContainsInvalidUnderscore(tok.val)
			if err != nil {
				p.raiseError(tok, "%s", err)
			}
			val, err = strconv.ParseInt(cleanedVal, 10, 64)
		}
		if err != nil {
			p.raiseError(tok, "%s", err)
		}
		return val
	case tokenFloat:
		err := numberContainsInvalidUnderscore(tok.val)
		if err != nil {
			p.raiseError(tok, "%s", err)
		}
		cleanedVal := cleanupNumberToken(tok.val)
		val, err := strconv.ParseFloat(cleanedVal, 64)
		if err != nil {
			p.raiseError(tok, "%s", err)
		}
		return val
	case tokenLocalTime:
		val, err := ParseLocalTime(tok.val)
		if err != nil {
			p.raiseError(tok, "%s", err)
		}
		return val
	case tokenLocalDate:
		// a local date may be followed by:
		// * nothing: this is a local date
		// * a local time: this is a local date-time

		next := p.peek()
		if next == nil || next.typ != tokenLocalTime {
			val, err := ParseLocalDate(tok.val)
			if err != nil {
				p.raiseError(tok, "%s", err)
			}
			return val
		}

		localDate := tok
		localTime := p.getToken()

		next = p.peek()
		if next == nil || next.typ != tokenTimeOffset {
			v := localDate.val + "T" + localTime.val
			val, err := ParseLocalDateTime(v)
			if err != nil {
				p.raiseError(tok, "%s", err)
			}
			return val
		}

		offset := p.getToken()

		layout := time.RFC3339Nano
		v := localDate.val + "T" + localTime.val + offset.val
		val, err := time.ParseInLocation(layout, v, time.UTC)
		if err != nil {
			p.raiseError(tok, "%s", err)
		}
		return val
	case tokenLeftBracket:
		return p.parseArray()
	case tokenLeftCurlyBrace:
		return p.parseInlineTable()
	case tokenEqual:
		p.raiseError(tok, "cannot have multiple equals for the same key")
	case tokenError:
		p.raiseError(tok, "%s", tok)
	default:
		panic(fmt.Errorf("unhandled token: %v", tok))
	}

	return nil
}

func tokenIsComma(t *token) bool {
	return t != nil && t.typ == tokenComma
}

func (p *tomlParser) parseInlineTable() *Tree {
	tree := newTree()
	var previous *token
Loop:
	for {
		follow := p.peek()
		if follow == nil || follow.typ == tokenEOF {
			p.raiseError(follow, "unterminated inline table")
		}
		switch follow.typ {
		case tokenRightCurlyBrace:
			p.getToken()
			break Loop
		case tokenKey, tokenInteger, tokenString:
			if !tokenIsComma(previous) && previous != nil {
				p.raiseError(follow, "comma expected between fields in inline table")
			}
			key := p.getToken()
			p.assume(tokenEqual)

			parsedKey, err := parseKey(key.val)
			if err != nil {
				p.raiseError(key, "invalid key: %s", err)
			}

			value := p.parseRvalue()
			tree.SetPath(parsedKey, value)
		case tokenComma:
			if tokenIsComma(previous) {
				p.raiseError(follow, "need field between two commas in inline table")
			}
			p.getToken()
		default:
			p.raiseError(follow, "unexpected token type in inline table: %s", follow.String())
		}
		previous = follow
	}
	if tokenIsComma(previous) {
		p.raiseError(previous, "trailing comma at the end of inline table")
	}
	tree.inline = true
	return tree
}

func (p *tomlParser) parseArray() interface{} {
	var array []interface{}
	arrayType := reflect.TypeOf(newTree())
	for {
		follow := p.peek()
		if follow == nil || follow.typ == tokenEOF {
			p.raiseError(follow, "unterminated array")
		}
		if follow.typ == tokenRightBracket {
			p.getToken()
			break
		}
		val := p.parseRvalue()
		if reflect.TypeOf(val) != arrayType {
			arrayType = nil
		}
		array = append(array, val)
		follow = p.peek()
		if follow == nil || follow.typ == tokenEOF {
			p.raiseError(follow, "unterminated array")
		}
		if follow.typ != tokenRightBracket && follow.typ != tokenComma {
			p.raiseError(follow, "missing comma")
		}
		if follow.typ == tokenComma {
			p.getToken()
		}
	}

	// if the array is a mixed-type array or its length is 0,
	// don't convert it to a table array
	if len(array) <= 0 {
		arrayType = nil
	}
	// An array of Trees is actually an array of inline
	// tables, which is a shorthand for a table array. If the
	// array was not converted from []interface{} to []*Tree,
	// the two notations would not be equivalent.
	if arrayType == reflect.TypeOf(newTree()) {
		tomlArray := make([]*Tree, len(array))
		for i, v := range array {
			tomlArray[i] = v.(*Tree)
		}
		return tomlArray
	}
	return array
}

func parseToml(flow []token) *Tree {
	result := newTree()
	result.position = Position{1, 1}
	parser := &tomlParser{
		flowIdx:       0,
		flow:          flow,
		tree:          result,
		currentTable:  make([]string, 0),
		seenTableKeys: make([]string, 0),
	}
	parser.run()
	return result
}
