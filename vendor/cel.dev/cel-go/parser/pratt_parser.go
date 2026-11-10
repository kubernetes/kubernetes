// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package parser

import (
	"fmt"
	"strconv"
	"strings"

	"cel.dev/cel-go/common"
	"cel.dev/cel-go/common/ast"
	"cel.dev/cel-go/common/operators"
	"cel.dev/cel-go/common/runes"
	"cel.dev/cel-go/common/types"
)

type binaryOpInfo struct {
	precedence int
	name       string
	kind       tokenKind
}

var (
	opLogicalOr        = binaryOpInfo{precedence: 1, name: operators.LogicalOr, kind: tokLogicalOr}
	opLogicalAnd       = binaryOpInfo{precedence: 2, name: operators.LogicalAnd, kind: tokLogicalAnd}
	opLess             = binaryOpInfo{precedence: 3, name: operators.Less, kind: tokLess}
	opLessEqual        = binaryOpInfo{precedence: 3, name: operators.LessEquals, kind: tokLessEqual}
	opGreater          = binaryOpInfo{precedence: 3, name: operators.Greater, kind: tokGreater}
	opGreaterEqual     = binaryOpInfo{precedence: 3, name: operators.GreaterEquals, kind: tokGreaterEqual}
	opEqualEqual       = binaryOpInfo{precedence: 3, name: operators.Equals, kind: tokEqualEqual}
	opExclamationEqual = binaryOpInfo{precedence: 3, name: operators.NotEquals, kind: tokExclamationEqual}
	opIn               = binaryOpInfo{precedence: 3, name: operators.In, kind: tokIn}
	opPlus             = binaryOpInfo{precedence: 4, name: operators.Add, kind: tokPlus}
	opMinus            = binaryOpInfo{precedence: 4, name: operators.Subtract, kind: tokMinus}
	opAsterisk         = binaryOpInfo{precedence: 5, name: operators.Multiply, kind: tokAsterisk}
	opSlash            = binaryOpInfo{precedence: 5, name: operators.Divide, kind: tokSlash}
	opPercent          = binaryOpInfo{precedence: 5, name: operators.Modulo, kind: tokPercent}
	opDefault          = binaryOpInfo{precedence: 0, name: "", kind: tokError}

	binaryOpInfoTable = [tokLogicalOr + 1]binaryOpInfo{
		tokLogicalOr:        opLogicalOr,
		tokLogicalAnd:       opLogicalAnd,
		tokLess:             opLess,
		tokLessEqual:        opLessEqual,
		tokGreater:          opGreater,
		tokGreaterEqual:     opGreaterEqual,
		tokEqualEqual:       opEqualEqual,
		tokExclamationEqual: opExclamationEqual,
		tokIn:               opIn,
		tokPlus:             opPlus,
		tokMinus:            opMinus,
		tokAsterisk:         opAsterisk,
		tokSlash:            opSlash,
		tokPercent:          opPercent,
	}
)

func getBinaryOpInfo(kind tokenKind) binaryOpInfo {
	if int(kind) < len(binaryOpInfoTable) {
		return binaryOpInfoTable[kind]
	}
	return opDefault
}

type prattParserWorker struct {
	content                    runes.Buffer
	length                     int32
	helper                     *parserHelper
	errors                     *parseErrors
	exprFactory                ast.ExprFactory
	lexer                      *lexer
	currTok                    token
	peekTok                    token
	macros                     map[string]Macro
	recursionDepth             int
	lastParsedDepth            int
	recursionLimitExceeded     bool
	errorCount                 int
	maxRecursionDepth          int
	maxExpressionNodeCount     int
	errorReportingLimit        int
	errorRecoveryLimit         int
	populateMacroCalls         bool
	enableOptionalSyntax       bool
	enableVariadicOperatorASTs bool
	enableIdentEscapeSyntax    bool
	enableCallEscapeSyntax     bool
}

// prattParser encapsulates the context necessary to perform Pratt parsing for different expressions.
type prattParser struct {
	options
}

// Parse parses the expression represented by source using the Pratt parser and returns the result.
func (p *prattParser) Parse(source common.Source) (*ast.AST, *common.Errors) {
	errs := common.NewErrors(source)
	buf, ok := source.(runes.Buffer)
	if !ok {
		buf = runes.NewBuffer(source.Content())
	}
	if buf.Len() > p.expressionSizeCodePointLimit {
		errs.ReportError(common.NoLocation,
			"expression code point size exceeds limit: size: %d, limit %d",
			buf.Len(), p.expressionSizeCodePointLimit)
		return nil, errs
	}
	accu := AccumulatorName
	if p.enableHiddenAccumulatorName {
		accu = HiddenAccumulatorName
	}
	fac := ast.NewExprFactoryWithAccumulator(accu)
	pratt := &prattParserWorker{
		content:                    buf,
		length:                     int32(buf.Len()),
		helper:                     newParserHelper(source, fac),
		errors:                     &parseErrors{errs},
		exprFactory:                fac,
		lexer:                      newLexer(buf),
		macros:                     p.macros,
		maxRecursionDepth:          p.maxRecursionDepth,
		maxExpressionNodeCount:     p.maxExpressionNodeCount,
		errorReportingLimit:        p.errorReportingLimit,
		errorRecoveryLimit:         p.errorRecoveryLimit,
		populateMacroCalls:         p.populateMacroCalls,
		enableOptionalSyntax:       p.enableOptionalSyntax,
		enableVariadicOperatorASTs: p.enableVariadicOperatorASTs,
		enableIdentEscapeSyntax:    p.enableIdentEscapeSyntax,
		enableCallEscapeSyntax:     p.enableCallEscapeSyntax,
	}
	pratt.initTokenStream()
	out := pratt.parse()
	if len(errs.GetErrors()) > 0 {
		return nil, errs
	}
	return ast.NewAST(out, pratt.helper.getSourceInfo()), errs
}

func (p *prattParserWorker) initTokenStream() {
	p.currTok = token{kind: tokError, start: 0, end: 0}
	p.peekTok = p.nextSignificantToken()
}

func (p *prattParserWorker) isRecoveryLimitExceeded() bool {
	return p.errorCount > p.errorRecoveryLimit
}

// checkRecursion returns true and trips the recursion limit if depth exceeds the configured
// limit.
//
// The parser bounds two depths separately against maxRecursionDepth:
//
//   - recursionDepth tracks syntactic expression nesting: each parenthesized expression, list,
//     map or struct element, call argument, index and conditional false branch opens a new
//     level. This bounds the parser's Go call-stack recursion (along with iterative runs of
//     grouping parentheses).
//   - lastParsedDepth tracks the depth of the AST parsed so far, counting binary, conditional,
//     field selection, member call and index nodes. Grouping parentheses, unary operators
//     (which are folded), logical operators (which are balanced or variadic), and composite or
//     call containers (lists, maps, structs and global calls, whose nesting is already bounded
//     by recursionDepth) do not add to it. Because operator and selector chains are parsed
//     iteratively, this depth is computed bottom-up and checked as each node is built.
func (p *prattParserWorker) checkRecursion(depth int) bool {
	if depth > p.maxRecursionDepth {
		if !p.recursionLimitExceeded {
			p.recursionLimitExceeded = true
			p.errors.internalError(fmt.Sprintf("expression recursion limit exceeded: %d", p.maxRecursionDepth))
			p.peekTok = token{kind: tokEnd, start: p.length, end: p.length}
		}
		return true
	}
	return false
}

func (p *prattParserWorker) nextSignificantToken() token {
	if p.isRecoveryLimitExceeded() {
		return token{kind: tokEnd, start: p.length, end: p.length}
	}
	for {
		tok := p.lexer.Lex()
		if tok.kind == tokWhitespace || tok.kind == tokComment {
			continue
		}
		if tok.kind == tokError {
			p.reportSyntaxError(tok, "%s", p.lexer.GetError().message)
			if p.isRecoveryLimitExceeded() {
				return token{kind: tokEnd, start: p.length, end: p.length}
			}
		}
		return tok
	}
}

func (p *prattParserWorker) scanNextSignificantToken() token {
	for {
		tok := p.lexer.Lex()
		if tok.kind == tokWhitespace || tok.kind == tokComment {
			continue
		}
		return tok
	}
}

// isStructCreationAhead checks whether the current identifier is the root of a struct/message
// creation expression (`CreateMessage` in grammar: `'.'? IDENTIFIER ('.' IDENTIFIER)* '{' ... '}'`).
// Unlike standalone identifiers or selector chains, reserved identifiers (e.g. `import{}` or `import.Foo{}`)
// are permitted in message type names.
func (p *prattParserWorker) isStructCreationAhead() bool {
	if p.peekTok.kind == tokLeftBrace {
		return true
	}
	if p.peekTok.kind != tokDot {
		return false
	}
	savedPos := p.lexer.SavePosition()
	defer p.lexer.RestorePosition(savedPos)

	tok := p.peekTok
	for tok.kind == tokDot {
		tok = p.scanNextSignificantToken()
		if tok.kind != tokIdent && tok.kind != tokReservedWord {
			return false
		}
		// Quoted identifiers are not allowed in struct creation expressions.
		text := p.tokenText(tok)
		if len(text) > 0 && text[0] == '`' {
			return false
		}
		tok = p.scanNextSignificantToken()
	}
	return tok.kind == tokLeftBrace
}

func (p *prattParserWorker) nextToken() token {
	p.currTok = p.peekTok
	if p.isRecoveryLimitExceeded() {
		p.peekTok = token{kind: tokEnd, start: p.length, end: p.length}
		return p.currTok
	}
	if p.peekTok.kind != tokEnd {
		p.peekTok = p.nextSignificantToken()
	}
	return p.currTok
}

func (p *prattParserWorker) tokenText(tok token) string {
	if tok.start >= 0 && tok.end >= tok.start && tok.end <= p.length {
		return p.content.Slice(int(tok.start), int(tok.end))
	}
	return ""
}

func (p *prattParserWorker) nextID(tok token) int64 {
	return p.helper.idFromOffsets(tok.start, tok.end)
}

func (p *prattParserWorker) expect(kind tokenKind, msg string) bool {
	if p.peekTok.kind == kind {
		p.nextToken()
		return true
	}
	if p.recursionLimitExceeded || p.isRecoveryLimitExceeded() {
		return false
	}
	if p.peekTok.kind != tokError {
		if msg == "" {
			tokText := p.tokenText(p.peekTok)
			formattedTok := fmt.Sprintf("'%s'", tokText)
			if p.peekTok.kind == tokEnd {
				formattedTok = "<EOF>"
			}
			msg = fmt.Sprintf("mismatched input %s expecting '%s'", formattedTok, kind.String())
		}
		p.reportSyntaxError(p.peekTok, "%s", msg)
	}
	p.synchronizeOnDelimiter()
	return false
}

func (p *prattParserWorker) synchronizeOnDelimiter() {
	if p.recursionLimitExceeded || p.isRecoveryLimitExceeded() {
		p.peekTok = token{kind: tokEnd, start: p.length, end: p.length}
		return
	}
	for p.peekTok.kind != tokEnd {
		if p.peekTok.kind == tokComma ||
			p.peekTok.kind == tokRightParen ||
			p.peekTok.kind == tokRightBracket ||
			p.peekTok.kind == tokRightBrace {
			break
		}
		p.nextToken()
	}
}

func (p *prattParserWorker) reportError(ctx any, format string, args ...any) ast.Expr {
	if p.isRecoveryLimitExceeded() {
		return p.helper.newExpr(common.NoLocation)
	}
	p.errorCount++
	var location common.Location
	err := p.helper.newExpr(ctx)
	switch c := ctx.(type) {
	case common.Location:
		location = c
	case token:
		location = p.helper.getLocation(err.ID())
	default:
		location = p.helper.getLocation(err.ID())
	}
	if p.errorCount <= p.errorReportingLimit {
		p.errors.reportErrorAtID(err.ID(), location, format, args...)
	}
	if p.isRecoveryLimitExceeded() {
		p.peekTok = token{kind: tokEnd, start: p.length, end: p.length}
	}
	return err
}

func (p *prattParserWorker) reportSyntaxError(ctx any, format string, args ...any) ast.Expr {
	return p.reportError(ctx, "Syntax error: "+format, args...)
}

func (p *prattParserWorker) newLogicManager(function string, term ast.Expr) *logicManager {
	if p.enableVariadicOperatorASTs {
		return newVariadicLogicManager(p.exprFactory, function, term)
	}
	return newBalancingLogicManager(p.exprFactory, function, term)
}

func (p *prattParserWorker) globalCallOrMacro(exprID int64, function string, args ...ast.Expr) ast.Expr {
	if expr, found := p.expandMacro(exprID, function, nil, args...); found {
		return expr
	}
	return p.helper.newGlobalCall(exprID, function, args...)
}

func (p *prattParserWorker) receiverCallOrMacro(exprID int64, function string, target ast.Expr, args ...ast.Expr) ast.Expr {
	if expr, found := p.expandMacro(exprID, function, target, args...); found {
		return expr
	}
	return p.helper.newReceiverCall(exprID, function, target, args...)
}

func (p *prattParserWorker) expandMacro(exprID int64, function string, target ast.Expr, args ...ast.Expr) (ast.Expr, bool) {
	if len(p.macros) == 0 {
		return nil, false
	}
	macro, found := p.macros[makeMacroKey(function, len(args), target != nil)]
	if !found {
		macro, found = p.macros[makeVarArgMacroKey(function, target != nil)]
		if !found {
			return nil, false
		}
	}
	if int(p.helper.expressionCount()) > p.maxExpressionNodeCount {
		loc := p.helper.getLocation(exprID)
		p.helper.deleteID(exprID)
		return p.reportError(loc, "expression count exceeds limit of %d while expanding macro '%s'", p.maxExpressionNodeCount, function), true
	}
	eh := exprHelperPool.Get().(*exprHelper)
	eh.parserHelper = p.helper
	eh.id = exprID
	expr, err := macro.Expander()(eh, target, args)
	exprHelperPool.Put(eh)
	if int(p.helper.expressionCount()) > p.maxExpressionNodeCount {
		loc := p.helper.getLocation(exprID)
		p.helper.deleteID(exprID)
		return p.reportError(loc, "expression count exceeds limit of %d while expanding macro '%s'", p.maxExpressionNodeCount, function), true
	}
	if err != nil {
		loc := err.Location
		if loc == nil {
			loc = p.helper.getLocation(exprID)
		}
		p.helper.deleteID(exprID)
		return p.reportError(loc, "%s", err.Message), true
	}
	if expr == nil {
		return nil, false
	}
	if p.populateMacroCalls {
		p.helper.addMacroCall(expr.ID(), function, target, args...)
	}
	p.helper.deleteID(exprID)
	return expr, true
}

func (p *prattParserWorker) normalizeIdent(tok token, allowQuoted bool, isQuoted *bool) string {
	if isQuoted != nil {
		*isQuoted = false
	}
	text := p.tokenText(tok)
	if len(text) == 0 {
		return ""
	}
	if text[0] == '`' {
		if isQuoted != nil {
			*isQuoted = true
		}
		if !allowQuoted {
			p.reportError(tok, "unexpected quoted identifier")
			return ""
		}
		if !p.enableIdentEscapeSyntax {
			p.reportError(tok, "unsupported syntax: '`'")
		}
		if len(text) < 2 || text[len(text)-1] != '`' {
			p.reportError(tok, "unterminated quoted identifier")
			return ""
		}
		// Validate the quoted identifier syntax:
		// ESC_IDENTIFIER : '`' (LETTER | DIGIT | '_' | '.' | '-' | '/' | ' ')+ '`';
		inner := text[1 : len(text)-1]
		if len(inner) == 0 {
			p.reportError(tok, "unexpected quoted identifier")
			return ""
		}
		for _, c := range inner {
			if !isAlpha(c) && !isDigit(c) && c != '_' && c != '.' && c != '-' && c != '/' && c != ' ' {
				if c == '@' && p.enableCallEscapeSyntax {
					continue
				}
				p.reportError(tok, "unexpected quoted identifier")
				return ""
			}
		}
		return inner
	}
	return text
}

func (p *prattParserWorker) parse() ast.Expr {
	expr := p.parseExpr()
	if p.recursionLimitExceeded || p.isRecoveryLimitExceeded() {
		return expr
	}
	if p.peekTok.kind != tokEnd && p.peekTok.kind != tokError {
		p.reportSyntaxError(p.peekTok, "unexpected token after expression")
	}
	return expr
}

func (p *prattParserWorker) parseExpr() ast.Expr {
	if p.recursionLimitExceeded || p.isRecoveryLimitExceeded() {
		return p.helper.newExpr(common.NoLocation)
	}
	if p.checkRecursion(p.recursionDepth + 1) {
		return p.helper.newExpr(common.NoLocation)
	}
	p.recursionDepth++
	expr := p.parseBinaryAndTernary(0)
	p.recursionDepth--
	return expr
}

// parseElementExpr parses one element of a delimited construct (list, map, struct or argument
// list), accumulating lastParsedDepth to the deepest element seen so far. A construct is as
// deep as its deepest element, not its last one, so callers must reset lastParsedDepth to 0
// before the first element.
func (p *prattParserWorker) parseElementExpr() ast.Expr {
	maxDepth := p.lastParsedDepth
	expr := p.parseExpr()
	if maxDepth > p.lastParsedDepth {
		p.lastParsedDepth = maxDepth
	}
	return expr
}

func (p *prattParserWorker) parseBinaryAndTernary(minPrec int) ast.Expr {
	lhs := p.parseSelectorChain()
	return p.parseBinaryAndTernaryFromLhs(lhs, minPrec, p.lastParsedDepth)
}

// parseBinaryAndTernaryFromLhs parses the operators following lhs, which is lhsDepth deep, and
// leaves the depth of the resulting expression in lastParsedDepth.
func (p *prattParserWorker) parseBinaryAndTernaryFromLhs(lhs ast.Expr, minPrec int, lhsDepth int) ast.Expr {
	depth := lhsDepth
	for !p.recursionLimitExceeded && !p.isRecoveryLimitExceeded() {
		tok := p.peekTok.kind
		if tok == tokQuestion && minPrec <= 0 {
			lhs = p.parseTernary(lhs, depth)
			depth = p.lastParsedDepth
			continue
		}

		opInfo := getBinaryOpInfo(tok)
		if opInfo.kind == tokError || opInfo.precedence < minPrec {
			break
		}

		if opInfo.name == operators.LogicalOr || opInfo.name == operators.LogicalAnd {
			lhs = p.parseLogicalChain(lhs, opInfo, depth)
			depth = p.lastParsedDepth
			continue
		}

		opTok := p.nextToken()
		opID := p.nextID(opTok)
		rhs := p.parseBinaryAndTernary(opInfo.precedence + 1)
		lhs = p.helper.newGlobalCall(opID, opInfo.name, lhs, rhs)
		// lastParsedDepth is the depth of the rhs just parsed. The operator node
		// sits one level above whichever operand is deeper: "x + a.b.c.d" reaches
		// 4 through its rhs and "a.b.c.d + x" reaches 4 through its lhs.
		depth = max(depth, p.lastParsedDepth) + 1
		p.checkRecursion(depth)
	}
	p.lastParsedDepth = depth
	return lhs
}

// parseTernary parses the branches of a conditional whose condition, lhs, is condDepth deep,
// and leaves the depth of the conditional in lastParsedDepth.
func (p *prattParserWorker) parseTernary(lhs ast.Expr, condDepth int) ast.Expr {
	qTok := p.nextToken()
	opID := p.nextID(qTok)
	trueExpr := p.parseBinaryAndTernary(1)
	depth := max(condDepth, p.lastParsedDepth)
	if !p.expect(tokColon, "expected ':' in conditional expression") {
		p.lastParsedDepth = condDepth
		return lhs
	}
	falseExpr := p.parseExpr()
	// The conditional node sits one level above the deepest of its operands.
	depth = max(depth, p.lastParsedDepth) + 1
	p.lastParsedDepth = depth
	p.checkRecursion(depth)
	return p.helper.newGlobalCall(opID, operators.Conditional, lhs, trueExpr, falseExpr)
}

// parseLogicalChain parses a run of '&&' or '||' operators following lhs, which is lhsDepth
// deep, and leaves the depth of the resulting expression in lastParsedDepth. Logical operators
// do not add to the depth: the chain is as deep as its deepest term.
func (p *prattParserWorker) parseLogicalChain(lhs ast.Expr, opInfo binaryOpInfo, lhsDepth int) ast.Expr {
	depth := lhsDepth
	l := p.newLogicManager(opInfo.name, lhs)
	for !p.recursionLimitExceeded && !p.isRecoveryLimitExceeded() && p.peekTok.kind == opInfo.kind {
		opTok := p.nextToken()
		rhs := p.parseBinaryAndTernary(opInfo.precedence + 1)
		depth = max(depth, p.lastParsedDepth)
		opID := p.nextID(opTok)
		l.addTerm(opID, rhs)
	}
	p.lastParsedDepth = depth
	return l.toExpr()
}

func (p *prattParserWorker) parseSelectorChain() ast.Expr {
	p.lastParsedDepth = 0
	tok := p.peekTok.kind
	if tok == tokExclamation || tok == tokMinus {
		return p.parseUnaryOps()
	}
	return p.parseMember()
}

func (p *prattParserWorker) parseMember() ast.Expr {
	p.lastParsedDepth = 0
	canBeStructName := false
	lhs := p.parsePrimary(&canBeStructName)
	return p.parseSelectorChainTail(lhs, canBeStructName, p.lastParsedDepth)
}

// parseSelectorChainTail parses the field selections, member calls, indexes and struct
// creations following lhs, which is lhsDepth deep, and leaves the depth of the resulting
// expression in lastParsedDepth.
func (p *prattParserWorker) parseSelectorChainTail(lhs ast.Expr, canBeStructName bool, lhsDepth int) ast.Expr {
	depth := lhsDepth
loop:
	for !p.recursionLimitExceeded {
		switch p.peekTok.kind {
		case tokDot:
			dotTok := p.nextToken()
			optional := false
			if p.peekTok.kind == tokQuestion {
				p.nextToken()
				optional = true
				if !p.enableOptionalSyntax {
					p.reportError(dotTok, "unsupported syntax '.?'")
				}
			}
			fieldTok := p.nextToken()
			if fieldTok.kind != tokIdent && fieldTok.kind != tokReservedWord {
				if fieldTok.kind != tokError {
					p.reportSyntaxError(fieldTok, "expected identifier after '.'")
				}
				p.synchronizeOnDelimiter()
				break loop
			}
			isMemberCall := p.peekTok.kind == tokLeftParen
			var isQuoted bool
			field := p.normalizeIdent(fieldTok, p.enableCallEscapeSyntax || !isMemberCall, &isQuoted)
			depth++
			if optional {
				fieldID := p.nextID(fieldTok)
				opID := p.nextID(dotTok)
				lhs = p.helper.newGlobalCall(opID, operators.OptSelect, lhs, p.helper.newLiteralString(fieldID, field))
				canBeStructName = false
			} else if isMemberCall {
				lparen := p.nextToken()
				callID := p.nextID(lparen)
				args := p.parseArguments(tokRightParen)
				lhs = p.receiverCallOrMacro(callID, field, lhs, args...)
				// parseArguments leaves lastParsedDepth at the deepest argument.
				// Arguments hang one level below the call node, so "a.f(b.c.d.e)" is 4
				// deep. The max preserves the selectors already walked when the
				// arguments are shallower, as in "a.b.c.f(1)".
				depth = max(depth, p.lastParsedDepth+1)
				canBeStructName = false
			} else {
				dotID := p.nextID(dotTok)
				lhs = p.helper.newSelect(dotID, lhs, field)
				canBeStructName = canBeStructName && !isQuoted
			}
			// A plain qualified name such as "a.b.Msg" may still turn out to be the
			// type name of a struct creation, which adds no depth, so its field
			// selections are only checked once the chain is known to be something else.
			if !canBeStructName {
				p.checkRecursion(depth)
			}
		case tokLeftBracket:
			bracketTok := p.nextToken()
			opID := p.nextID(bracketTok)
			optional := false
			if p.peekTok.kind == tokQuestion {
				p.nextToken()
				optional = true
				if !p.enableOptionalSyntax {
					p.reportError(bracketTok, "unsupported syntax '[?'")
				}
			}
			index := p.parseExpr()
			p.expect(tokRightBracket, "expected ']'")
			opName := operators.Index
			if optional {
				opName = operators.OptIndex
			}
			lhs = p.helper.newGlobalCall(opID, opName, lhs, index)
			// lastParsedDepth is the depth of the index expression just parsed. The
			// index node sits one level above the deeper of its operand and its index,
			// so "a[b.c.d.e]" is 4 deep and "a.b.c[0]" is 3 deep.
			depth = max(depth, p.lastParsedDepth) + 1
			canBeStructName = false
			p.checkRecursion(depth)
		case tokLeftBrace:
			if !canBeStructName {
				break loop
			}
			structName, ok := p.extractStructName(lhs)
			if !ok {
				break loop
			}
			objID := p.nextID(p.peekTok)
			lhs = p.parseStruct(objID, structName)
			// parseStruct leaves lastParsedDepth at the deepest field value. Struct
			// creation is a primary rather than a chain link and its qualified type
			// name is part of it, so neither adds a level: "a.b.Msg{f: 1}" is 0 deep
			// and "Msg{f: a.b.c.d}" is 3 deep.
			depth = p.lastParsedDepth
			canBeStructName = false
		default:
			break loop
		}
	}
	// A chain that ended as a plain qualified name, such as "a.b.c", still needs
	// to be checked.
	if canBeStructName {
		p.checkRecursion(depth)
	}
	p.lastParsedDepth = depth
	return lhs
}

// extractStructName converts a chain of pre-parsed AST nodes representing field selections,
// such as "a.b.Msg", into a qualified struct type name.
func (p *prattParserWorker) extractStructName(expr ast.Expr) (string, bool) {
	var name string
	for expr != nil && expr.Kind() == ast.SelectKind {
		sel := expr.AsSelect()
		if sel.IsTestOnly() {
			return "", false
		}
		p.helper.deleteID(expr.ID())
		name = "." + sel.FieldName() + name
		expr = sel.Operand()
	}
	if expr == nil || expr.Kind() != ast.IdentKind {
		return "", false
	}
	p.helper.deleteID(expr.ID())
	return expr.AsIdent() + name, true
}

func (p *prattParserWorker) parseStruct(objID int64, structName string) ast.Expr {
	p.nextToken() // consumes {
	var fields []ast.EntryExpr
	p.lastParsedDepth = 0
	for p.peekTok.kind != tokRightBrace && p.peekTok.kind != tokEnd {
		optional := false
		if p.peekTok.kind == tokQuestion {
			q := p.nextToken()
			optional = true
			if !p.enableOptionalSyntax {
				p.reportError(q, "unsupported syntax '?'")
			}
		}
		fieldTok := p.nextToken()
		if fieldTok.kind != tokIdent && fieldTok.kind != tokReservedWord {
			p.reportSyntaxError(fieldTok, "expected struct field name")
			p.synchronizeOnDelimiter()
			break
		}
		fieldName := p.normalizeIdent(fieldTok, true, nil)
		colonTok := p.peekTok
		if !p.expect(tokColon, "expected ':' in struct field") {
			break
		}
		fieldID := p.nextID(colonTok)
		val := p.parseElementExpr()
		fields = append(fields, p.helper.newObjectField(fieldID, fieldName, val, optional))
		if p.peekTok.kind == tokComma {
			p.nextToken()
		} else {
			break
		}
	}
	p.expect(tokRightBrace, "expected '}'")
	return p.helper.newObject(objID, structName, fields...)
}

func (p *prattParserWorker) parseUnaryOps() ast.Expr {
	firstOp := p.nextToken()
	opType := firstOp.kind
	ops := []token{firstOp}
	for p.peekTok.kind == opType {
		ops = append(ops, p.nextToken())
	}

	if opType == tokMinus && len(ops) == 1 && (p.peekTok.kind == tokInt || p.peekTok.kind == tokFloat) {
		opID := p.nextID(firstOp)
		var lhs ast.Expr
		if p.peekTok.kind == tokInt {
			lhs = p.parseNegativeIntLiteral(opID)
		} else {
			lhs = p.parseNegativeDoubleLiteral(opID)
		}
		tok := p.peekTok.kind
		if tok == tokDot || tok == tokLeftBracket || tok == tokLeftBrace {
			lhs = p.parseSelectorChainTail(lhs, false, 0)
		}
		return lhs
	}

	if len(ops)%2 == 0 {
		ops = ops[:0]
	} else {
		ops = ops[:1]
	}

	type unaryOpWithID struct {
		token token
		id    int64
	}
	retainedOps := make([]unaryOpWithID, len(ops))
	for i, op := range ops {
		retainedOps[i] = unaryOpWithID{token: op, id: p.nextID(op)}
	}

	var operand ast.Expr
	if opType == tokExclamation && p.peekTok.kind == tokMinus {
		minusTok := p.nextToken()
		minusID := p.nextID(minusTok)
		if p.peekTok.kind == tokInt {
			operand = p.parseNegativeIntLiteral(minusID)
			operand = p.parseSelectorChainTail(operand, false, 0)
		} else if p.peekTok.kind == tokFloat {
			operand = p.parseNegativeDoubleLiteral(minusID)
			operand = p.parseSelectorChainTail(operand, false, 0)
		} else {
			p.reportSyntaxError(minusTok, "unexpected '-'")
			operand = p.parseMember()
		}
	} else {
		operand = p.parseMember()
	}
	if p.recursionLimitExceeded {
		return p.helper.newExpr(common.NoLocation)
	}

	for i := len(retainedOps) - 1; i >= 0; i-- {
		opName := operators.LogicalNot
		if retainedOps[i].token.kind == tokMinus {
			opName = operators.Negate
		}
		operand = p.globalCallOrMacro(retainedOps[i].id, opName, operand)
	}
	return operand
}

func (p *prattParserWorker) parsePrimary(canBeStructName *bool) ast.Expr {
	if canBeStructName != nil {
		*canBeStructName = false
	}
	switch p.peekTok.kind {
	case tokLeftParen:
		if p.recursionLimitExceeded || p.isRecoveryLimitExceeded() {
			return p.helper.newExpr(common.NoLocation)
		}
		// To avoid deep call-stack recursion on heavily nested parentheses (e.g.
		// "((((a))))" or "((((a + 1) + 1) + 1))"), consume all consecutive
		// leading '(' tokens upfront, parse the innermost expression once, and
		// then iteratively unwind each enclosing '(' from innermost to outermost.
		// After consuming each matching ')', if more enclosing '(' remain open
		// and the next token is not another ')', continue parsing any trailing
		// selectors or binary/ternary operators belonging to that enclosing
		// parenthesized level using the already-parsed inner expression as the
		// LHS.
		openParens := 0
		for p.peekTok.kind == tokLeftParen {
			openParens++
			p.nextToken()
		}
		// Every '(' nested inside the enclosing expression is a nesting level and
		// costs one unit of recursion budget, so charge all of them here: the
		// innermost of openParens parentheses sits at recursionDepth + openParens - 1.
		// recursionDepth itself only advances by 1 because the parens are unwound
		// iteratively and entering parseBinaryAndTernary(0) below adds just one
		// stack frame. Parentheses do not add to the depth of the AST.
		if p.checkRecursion(p.recursionDepth + openParens - 1) {
			return p.helper.newExpr(common.NoLocation)
		}
		p.recursionDepth++
		expr := p.parseBinaryAndTernary(0)
		depth := p.lastParsedDepth
		for i := 0; i < openParens; i++ {
			p.expect(tokRightParen, "expected ')'")
			if i < openParens-1 && p.peekTok.kind != tokRightParen {
				tok := p.peekTok.kind
				if tok == tokDot || tok == tokLeftBracket || tok == tokLeftBrace {
					expr = p.parseSelectorChainTail(expr, false, depth)
					depth = p.lastParsedDepth
				}
				expr = p.parseBinaryAndTernaryFromLhs(expr, 0, depth)
				depth = p.lastParsedDepth
			}
		}
		p.recursionDepth--
		p.lastParsedDepth = depth
		return expr
	case tokNull:
		return p.helper.exprFactory.NewLiteral(p.nextID(p.nextToken()), types.NullValue)
	case tokTrue:
		tok := p.nextToken()
		return p.helper.newLiteralBool(p.nextID(tok), true)
	case tokFalse:
		tok := p.nextToken()
		return p.helper.newLiteralBool(p.nextID(tok), false)
	case tokInt:
		return p.parseIntLiteral()
	case tokUint:
		return p.parseUintLiteral()
	case tokFloat:
		return p.parseDoubleLiteral()
	case tokString:
		return p.parseStringLiteral()
	case tokBytes:
		return p.parseBytesLiteral()
	case tokLeftBracket:
		return p.parseList()
	case tokLeftBrace:
		return p.parseMap()
	case tokDot, tokIdent, tokReservedWord:
		return p.parseIdentOrCall(canBeStructName)
	default:
		badTok := p.nextToken()
		if badTok.kind != tokError {
			if badTok.kind == tokEnd {
				p.reportSyntaxError(badTok, "mismatched input '<EOF>' expecting expression")
			} else {
				p.reportSyntaxError(badTok, "unexpected token")
			}
		}
		return p.helper.newExpr(badTok)
	}
}

func (p *prattParserWorker) parseList() ast.Expr {
	openTok := p.nextToken()
	listID := p.nextID(openTok)
	var elems []ast.Expr
	var optionals []int32
	p.lastParsedDepth = 0
	for p.peekTok.kind != tokRightBracket && p.peekTok.kind != tokEnd {
		optional := false
		if p.peekTok.kind == tokQuestion {
			q := p.nextToken()
			optional = true
			if !p.enableOptionalSyntax {
				p.reportError(q, "unsupported syntax '?'")
			}
		}
		if optional {
			optionals = append(optionals, int32(len(elems)))
		}
		elem := p.parseElementExpr()
		elems = append(elems, elem)
		if p.peekTok.kind == tokComma {
			p.nextToken()
			if p.peekTok.kind == tokRightBracket {
				break
			}
			continue
		}
		break
	}
	p.expect(tokRightBracket, "expected ']'")
	return p.helper.newList(listID, elems, optionals...)
}

func (p *prattParserWorker) parseMap() ast.Expr {
	openTok := p.nextToken()
	mapID := p.nextID(openTok)
	var entries []ast.EntryExpr
	p.lastParsedDepth = 0
	for p.peekTok.kind != tokRightBrace && p.peekTok.kind != tokEnd {
		optional := false
		if p.peekTok.kind == tokQuestion {
			q := p.nextToken()
			optional = true
			if !p.enableOptionalSyntax {
				p.reportError(q, "unsupported syntax '?'")
			}
		}
		entryID := p.helper.allocID()
		key := p.parseElementExpr()
		colonTok := p.peekTok
		if !p.expect(tokColon, "expected ':' in map entry") {
			break
		}
		p.helper.setTokenLocation(entryID, colonTok)
		val := p.parseElementExpr()
		entries = append(entries, p.helper.newMapEntry(entryID, key, val, optional))
		if p.peekTok.kind == tokComma {
			p.nextToken()
			if p.peekTok.kind == tokRightBrace {
				break
			}
			continue
		}
		break
	}
	p.expect(tokRightBrace, "expected '}'")
	return p.helper.newMap(mapID, entries...)
}

func (p *prattParserWorker) parseIdentOrCall(canBeStructName *bool) ast.Expr {
	leadingDot := false
	firstTok := p.peekTok
	if p.peekTok.kind == tokDot {
		p.nextToken()
		leadingDot = true
	}
	idTok := p.nextToken()
	if idTok.kind != tokIdent && idTok.kind != tokReservedWord {
		if idTok.kind != tokError {
			p.reportSyntaxError(idTok, "expected identifier")
		}
		if canBeStructName != nil {
			*canBeStructName = false
		}
		return p.helper.newExpr(idTok)
	}
	var isQuoted bool
	idText := p.normalizeIdent(idTok, p.enableCallEscapeSyntax, &isQuoted)
	if idTok.kind == tokReservedWord {
		if _, ok := reservedIds[idText]; ok && !p.isStructCreationAhead() {
			p.reportError(idTok, "reserved identifier: %s", idText)
		}
	}
	name := idText
	if leadingDot {
		name = "." + idText
	}
	if p.peekTok.kind == tokLeftParen {
		lparen := p.nextToken()
		callID := p.nextID(lparen)
		args := p.parseArguments(tokRightParen)
		if canBeStructName != nil {
			*canBeStructName = false
		}
		return p.globalCallOrMacro(callID, name, args...)
	}
	if canBeStructName != nil {
		*canBeStructName = !isQuoted
	}
	targetTok := idTok
	if leadingDot {
		targetTok = firstTok
	}
	id := p.nextID(targetTok)
	return p.helper.newIdent(id, name)
}

func (p *prattParserWorker) parseArguments(closeTok tokenKind) []ast.Expr {
	var args []ast.Expr
	p.lastParsedDepth = 0
	if p.peekTok.kind != closeTok && p.peekTok.kind != tokEnd {
		for {
			args = append(args, p.parseElementExpr())
			if p.peekTok.kind == tokComma {
				p.nextToken()
				if p.peekTok.kind == closeTok {
					p.reportSyntaxError(p.peekTok, "unexpected token")
					break
				}
				continue
			}
			break
		}
	}
	p.expect(closeTok, "")
	return args
}

func (p *prattParserWorker) parseIntLiteral() ast.Expr {
	tok := p.nextToken()
	id := p.nextID(tok)
	text := p.tokenText(tok)
	base := 10
	if strings.HasPrefix(text, "0x") || strings.HasPrefix(text, "0X") {
		base = 16
		text = text[2:]
	}
	val, err := strconv.ParseInt(text, base, 64)
	if err != nil {
		return p.reportSyntaxError(tok, "invalid int literal")
	}
	return p.helper.newLiteralInt(id, val)
}

func (p *prattParserWorker) parseNegativeIntLiteral(opID int64) ast.Expr {
	tok := p.nextToken()
	text := p.tokenText(tok)
	base := 10
	if strings.HasPrefix(text, "0x") || strings.HasPrefix(text, "0X") {
		base = 16
		text = text[2:]
	}
	val, err := strconv.ParseInt("-"+text, base, 64)
	if err != nil {
		return p.reportSyntaxError(tok, "invalid int literal")
	}
	return p.helper.newLiteralInt(opID, val)
}

func (p *prattParserWorker) parseUintLiteral() ast.Expr {
	tok := p.nextToken()
	id := p.nextID(tok)
	text := p.tokenText(tok)
	text = text[:len(text)-1]
	base := 10
	if strings.HasPrefix(text, "0x") || strings.HasPrefix(text, "0X") {
		base = 16
		text = text[2:]
	}
	val, err := strconv.ParseUint(text, base, 64)
	if err != nil {
		return p.reportSyntaxError(tok, "invalid uint literal")
	}
	return p.helper.newLiteralUint(id, val)
}

func (p *prattParserWorker) parseDoubleLiteral() ast.Expr {
	tok := p.nextToken()
	id := p.nextID(tok)
	text := p.tokenText(tok)
	val, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return p.reportSyntaxError(tok, "invalid double literal")
	}
	return p.helper.newLiteralDouble(id, val)
}

func (p *prattParserWorker) parseNegativeDoubleLiteral(opID int64) ast.Expr {
	tok := p.nextToken()
	text := p.tokenText(tok)
	val, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return p.reportSyntaxError(tok, "invalid double literal")
	}
	return p.helper.newLiteralDouble(opID, -val)
}

func (p *prattParserWorker) parseStringLiteral() ast.Expr {
	tok := p.nextToken()
	id := p.nextID(tok)
	text := p.tokenText(tok)
	unescaped, err := unescape(text, false)
	if err != nil {
		return p.reportError(tok, "%s", err.Error())
	}
	return p.helper.newLiteralString(id, unescaped)
}

func (p *prattParserWorker) parseBytesLiteral() ast.Expr {
	tok := p.nextToken()
	id := p.nextID(tok)
	text := p.tokenText(tok)
	if strings.HasPrefix(text, "b") || strings.HasPrefix(text, "B") {
		text = text[1:]
	} else if strings.HasPrefix(text, "rb") || strings.HasPrefix(text, "RB") || strings.HasPrefix(text, "rB") || strings.HasPrefix(text, "Rb") {
		text = "r" + text[2:]
	} else if strings.HasPrefix(text, "br") || strings.HasPrefix(text, "BR") || strings.HasPrefix(text, "bR") || strings.HasPrefix(text, "Br") {
		text = text[1:]
	}
	unescaped, err := unescape(text, true)
	if err != nil {
		return p.reportError(tok, "%s", err.Error())
	}
	return p.helper.newLiteralBytes(id, []byte(unescaped))
}
