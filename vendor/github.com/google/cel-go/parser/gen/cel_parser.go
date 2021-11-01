// Generated from /Users/tswadell/go/src/github.com/google/cel-go/bin/../parser/gen/CEL.g4 by ANTLR 4.7.

package gen // CEL
import (
	"fmt"
	"reflect"
	"strconv"

	"github.com/antlr/antlr4/runtime/Go/antlr"
)

// Suppress unused import errors
var _ = fmt.Printf
var _ = reflect.Copy
var _ = strconv.Itoa

var parserATN = []uint16{
	3, 24715, 42794, 33075, 47597, 16764, 15335, 30598, 22884, 3, 38, 211,
	4, 2, 9, 2, 4, 3, 9, 3, 4, 4, 9, 4, 4, 5, 9, 5, 4, 6, 9, 6, 4, 7, 9, 7,
	4, 8, 9, 8, 4, 9, 9, 9, 4, 10, 9, 10, 4, 11, 9, 11, 4, 12, 9, 12, 4, 13,
	9, 13, 4, 14, 9, 14, 3, 2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 5, 3, 38, 10, 3, 3, 4, 3, 4, 3, 4, 7, 4, 43, 10, 4, 12, 4, 14, 4, 46,
	11, 4, 3, 5, 3, 5, 3, 5, 7, 5, 51, 10, 5, 12, 5, 14, 5, 54, 11, 5, 3, 6,
	3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 7, 6, 62, 10, 6, 12, 6, 14, 6, 65, 11, 6,
	3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 7, 7, 76, 10, 7,
	12, 7, 14, 7, 79, 11, 7, 3, 8, 3, 8, 6, 8, 83, 10, 8, 13, 8, 14, 8, 84,
	3, 8, 3, 8, 6, 8, 89, 10, 8, 13, 8, 14, 8, 90, 3, 8, 5, 8, 94, 10, 8, 3,
	9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 5, 9, 104, 10, 9, 3, 9, 5,
	9, 107, 10, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 5, 9, 117,
	10, 9, 3, 9, 5, 9, 120, 10, 9, 3, 9, 7, 9, 123, 10, 9, 12, 9, 14, 9, 126,
	11, 9, 3, 10, 5, 10, 129, 10, 10, 3, 10, 3, 10, 3, 10, 5, 10, 134, 10,
	10, 3, 10, 5, 10, 137, 10, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10,
	5, 10, 145, 10, 10, 3, 10, 5, 10, 148, 10, 10, 3, 10, 3, 10, 3, 10, 5,
	10, 153, 10, 10, 3, 10, 5, 10, 156, 10, 10, 3, 10, 3, 10, 5, 10, 160, 10,
	10, 3, 11, 3, 11, 3, 11, 7, 11, 165, 10, 11, 12, 11, 14, 11, 168, 11, 11,
	3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 7, 12, 177, 10, 12, 12,
	12, 14, 12, 180, 11, 12, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13,
	3, 13, 7, 13, 190, 10, 13, 12, 13, 14, 13, 193, 11, 13, 3, 14, 5, 14, 196,
	10, 14, 3, 14, 3, 14, 3, 14, 5, 14, 201, 10, 14, 3, 14, 3, 14, 3, 14, 3,
	14, 3, 14, 3, 14, 5, 14, 209, 10, 14, 3, 14, 2, 5, 10, 12, 16, 15, 2, 4,
	6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 2, 5, 3, 2, 3, 9, 3, 2, 25, 27,
	4, 2, 20, 20, 24, 24, 2, 237, 2, 28, 3, 2, 2, 2, 4, 31, 3, 2, 2, 2, 6,
	39, 3, 2, 2, 2, 8, 47, 3, 2, 2, 2, 10, 55, 3, 2, 2, 2, 12, 66, 3, 2, 2,
	2, 14, 93, 3, 2, 2, 2, 16, 95, 3, 2, 2, 2, 18, 159, 3, 2, 2, 2, 20, 161,
	3, 2, 2, 2, 22, 169, 3, 2, 2, 2, 24, 181, 3, 2, 2, 2, 26, 208, 3, 2, 2,
	2, 28, 29, 5, 4, 3, 2, 29, 30, 7, 2, 2, 3, 30, 3, 3, 2, 2, 2, 31, 37, 5,
	6, 4, 2, 32, 33, 7, 22, 2, 2, 33, 34, 5, 6, 4, 2, 34, 35, 7, 23, 2, 2,
	35, 36, 5, 4, 3, 2, 36, 38, 3, 2, 2, 2, 37, 32, 3, 2, 2, 2, 37, 38, 3,
	2, 2, 2, 38, 5, 3, 2, 2, 2, 39, 44, 5, 8, 5, 2, 40, 41, 7, 11, 2, 2, 41,
	43, 5, 8, 5, 2, 42, 40, 3, 2, 2, 2, 43, 46, 3, 2, 2, 2, 44, 42, 3, 2, 2,
	2, 44, 45, 3, 2, 2, 2, 45, 7, 3, 2, 2, 2, 46, 44, 3, 2, 2, 2, 47, 52, 5,
	10, 6, 2, 48, 49, 7, 10, 2, 2, 49, 51, 5, 10, 6, 2, 50, 48, 3, 2, 2, 2,
	51, 54, 3, 2, 2, 2, 52, 50, 3, 2, 2, 2, 52, 53, 3, 2, 2, 2, 53, 9, 3, 2,
	2, 2, 54, 52, 3, 2, 2, 2, 55, 56, 8, 6, 1, 2, 56, 57, 5, 12, 7, 2, 57,
	63, 3, 2, 2, 2, 58, 59, 12, 3, 2, 2, 59, 60, 9, 2, 2, 2, 60, 62, 5, 10,
	6, 4, 61, 58, 3, 2, 2, 2, 62, 65, 3, 2, 2, 2, 63, 61, 3, 2, 2, 2, 63, 64,
	3, 2, 2, 2, 64, 11, 3, 2, 2, 2, 65, 63, 3, 2, 2, 2, 66, 67, 8, 7, 1, 2,
	67, 68, 5, 14, 8, 2, 68, 77, 3, 2, 2, 2, 69, 70, 12, 4, 2, 2, 70, 71, 9,
	3, 2, 2, 71, 76, 5, 12, 7, 5, 72, 73, 12, 3, 2, 2, 73, 74, 9, 4, 2, 2,
	74, 76, 5, 12, 7, 4, 75, 69, 3, 2, 2, 2, 75, 72, 3, 2, 2, 2, 76, 79, 3,
	2, 2, 2, 77, 75, 3, 2, 2, 2, 77, 78, 3, 2, 2, 2, 78, 13, 3, 2, 2, 2, 79,
	77, 3, 2, 2, 2, 80, 94, 5, 16, 9, 2, 81, 83, 7, 21, 2, 2, 82, 81, 3, 2,
	2, 2, 83, 84, 3, 2, 2, 2, 84, 82, 3, 2, 2, 2, 84, 85, 3, 2, 2, 2, 85, 86,
	3, 2, 2, 2, 86, 94, 5, 16, 9, 2, 87, 89, 7, 20, 2, 2, 88, 87, 3, 2, 2,
	2, 89, 90, 3, 2, 2, 2, 90, 88, 3, 2, 2, 2, 90, 91, 3, 2, 2, 2, 91, 92,
	3, 2, 2, 2, 92, 94, 5, 16, 9, 2, 93, 80, 3, 2, 2, 2, 93, 82, 3, 2, 2, 2,
	93, 88, 3, 2, 2, 2, 94, 15, 3, 2, 2, 2, 95, 96, 8, 9, 1, 2, 96, 97, 5,
	18, 10, 2, 97, 124, 3, 2, 2, 2, 98, 99, 12, 5, 2, 2, 99, 100, 7, 18, 2,
	2, 100, 106, 7, 38, 2, 2, 101, 103, 7, 16, 2, 2, 102, 104, 5, 20, 11, 2,
	103, 102, 3, 2, 2, 2, 103, 104, 3, 2, 2, 2, 104, 105, 3, 2, 2, 2, 105,
	107, 7, 17, 2, 2, 106, 101, 3, 2, 2, 2, 106, 107, 3, 2, 2, 2, 107, 123,
	3, 2, 2, 2, 108, 109, 12, 4, 2, 2, 109, 110, 7, 12, 2, 2, 110, 111, 5,
	4, 3, 2, 111, 112, 7, 13, 2, 2, 112, 123, 3, 2, 2, 2, 113, 114, 12, 3,
	2, 2, 114, 116, 7, 14, 2, 2, 115, 117, 5, 22, 12, 2, 116, 115, 3, 2, 2,
	2, 116, 117, 3, 2, 2, 2, 117, 119, 3, 2, 2, 2, 118, 120, 7, 19, 2, 2, 119,
	118, 3, 2, 2, 2, 119, 120, 3, 2, 2, 2, 120, 121, 3, 2, 2, 2, 121, 123,
	7, 15, 2, 2, 122, 98, 3, 2, 2, 2, 122, 108, 3, 2, 2, 2, 122, 113, 3, 2,
	2, 2, 123, 126, 3, 2, 2, 2, 124, 122, 3, 2, 2, 2, 124, 125, 3, 2, 2, 2,
	125, 17, 3, 2, 2, 2, 126, 124, 3, 2, 2, 2, 127, 129, 7, 18, 2, 2, 128,
	127, 3, 2, 2, 2, 128, 129, 3, 2, 2, 2, 129, 130, 3, 2, 2, 2, 130, 136,
	7, 38, 2, 2, 131, 133, 7, 16, 2, 2, 132, 134, 5, 20, 11, 2, 133, 132, 3,
	2, 2, 2, 133, 134, 3, 2, 2, 2, 134, 135, 3, 2, 2, 2, 135, 137, 7, 17, 2,
	2, 136, 131, 3, 2, 2, 2, 136, 137, 3, 2, 2, 2, 137, 160, 3, 2, 2, 2, 138,
	139, 7, 16, 2, 2, 139, 140, 5, 4, 3, 2, 140, 141, 7, 17, 2, 2, 141, 160,
	3, 2, 2, 2, 142, 144, 7, 12, 2, 2, 143, 145, 5, 20, 11, 2, 144, 143, 3,
	2, 2, 2, 144, 145, 3, 2, 2, 2, 145, 147, 3, 2, 2, 2, 146, 148, 7, 19, 2,
	2, 147, 146, 3, 2, 2, 2, 147, 148, 3, 2, 2, 2, 148, 149, 3, 2, 2, 2, 149,
	160, 7, 13, 2, 2, 150, 152, 7, 14, 2, 2, 151, 153, 5, 24, 13, 2, 152, 151,
	3, 2, 2, 2, 152, 153, 3, 2, 2, 2, 153, 155, 3, 2, 2, 2, 154, 156, 7, 19,
	2, 2, 155, 154, 3, 2, 2, 2, 155, 156, 3, 2, 2, 2, 156, 157, 3, 2, 2, 2,
	157, 160, 7, 15, 2, 2, 158, 160, 5, 26, 14, 2, 159, 128, 3, 2, 2, 2, 159,
	138, 3, 2, 2, 2, 159, 142, 3, 2, 2, 2, 159, 150, 3, 2, 2, 2, 159, 158,
	3, 2, 2, 2, 160, 19, 3, 2, 2, 2, 161, 166, 5, 4, 3, 2, 162, 163, 7, 19,
	2, 2, 163, 165, 5, 4, 3, 2, 164, 162, 3, 2, 2, 2, 165, 168, 3, 2, 2, 2,
	166, 164, 3, 2, 2, 2, 166, 167, 3, 2, 2, 2, 167, 21, 3, 2, 2, 2, 168, 166,
	3, 2, 2, 2, 169, 170, 7, 38, 2, 2, 170, 171, 7, 23, 2, 2, 171, 178, 5,
	4, 3, 2, 172, 173, 7, 19, 2, 2, 173, 174, 7, 38, 2, 2, 174, 175, 7, 23,
	2, 2, 175, 177, 5, 4, 3, 2, 176, 172, 3, 2, 2, 2, 177, 180, 3, 2, 2, 2,
	178, 176, 3, 2, 2, 2, 178, 179, 3, 2, 2, 2, 179, 23, 3, 2, 2, 2, 180, 178,
	3, 2, 2, 2, 181, 182, 5, 4, 3, 2, 182, 183, 7, 23, 2, 2, 183, 191, 5, 4,
	3, 2, 184, 185, 7, 19, 2, 2, 185, 186, 5, 4, 3, 2, 186, 187, 7, 23, 2,
	2, 187, 188, 5, 4, 3, 2, 188, 190, 3, 2, 2, 2, 189, 184, 3, 2, 2, 2, 190,
	193, 3, 2, 2, 2, 191, 189, 3, 2, 2, 2, 191, 192, 3, 2, 2, 2, 192, 25, 3,
	2, 2, 2, 193, 191, 3, 2, 2, 2, 194, 196, 7, 20, 2, 2, 195, 194, 3, 2, 2,
	2, 195, 196, 3, 2, 2, 2, 196, 197, 3, 2, 2, 2, 197, 209, 7, 34, 2, 2, 198,
	209, 7, 35, 2, 2, 199, 201, 7, 20, 2, 2, 200, 199, 3, 2, 2, 2, 200, 201,
	3, 2, 2, 2, 201, 202, 3, 2, 2, 2, 202, 209, 7, 33, 2, 2, 203, 209, 7, 36,
	2, 2, 204, 209, 7, 37, 2, 2, 205, 209, 7, 28, 2, 2, 206, 209, 7, 29, 2,
	2, 207, 209, 7, 30, 2, 2, 208, 195, 3, 2, 2, 2, 208, 198, 3, 2, 2, 2, 208,
	200, 3, 2, 2, 2, 208, 203, 3, 2, 2, 2, 208, 204, 3, 2, 2, 2, 208, 205,
	3, 2, 2, 2, 208, 206, 3, 2, 2, 2, 208, 207, 3, 2, 2, 2, 209, 27, 3, 2,
	2, 2, 31, 37, 44, 52, 63, 75, 77, 84, 90, 93, 103, 106, 116, 119, 122,
	124, 128, 133, 136, 144, 147, 152, 155, 159, 166, 178, 191, 195, 200, 208,
}

var literalNames = []string{
	"", "'=='", "'!='", "'in'", "'<'", "'<='", "'>='", "'>'", "'&&'", "'||'",
	"'['", "']'", "'{'", "'}'", "'('", "')'", "'.'", "','", "'-'", "'!'", "'?'",
	"':'", "'+'", "'*'", "'/'", "'%'", "'true'", "'false'", "'null'",
}
var symbolicNames = []string{
	"", "EQUALS", "NOT_EQUALS", "IN", "LESS", "LESS_EQUALS", "GREATER_EQUALS",
	"GREATER", "LOGICAL_AND", "LOGICAL_OR", "LBRACKET", "RPRACKET", "LBRACE",
	"RBRACE", "LPAREN", "RPAREN", "DOT", "COMMA", "MINUS", "EXCLAM", "QUESTIONMARK",
	"COLON", "PLUS", "STAR", "SLASH", "PERCENT", "TRUE", "FALSE", "NULL", "WHITESPACE",
	"COMMENT", "NUM_FLOAT", "NUM_INT", "NUM_UINT", "STRING", "BYTES", "IDENTIFIER",
}

var ruleNames = []string{
	"start", "expr", "conditionalOr", "conditionalAnd", "relation", "calc",
	"unary", "member", "primary", "exprList", "fieldInitializerList", "mapInitializerList",
	"literal",
}

type CELParser struct {
	*antlr.BaseParser
}

func NewCELParser(input antlr.TokenStream) *CELParser {
	this := new(CELParser)
	deserializer := antlr.NewATNDeserializer(nil)
	deserializedATN := deserializer.DeserializeFromUInt16(parserATN)
	decisionToDFA := make([]*antlr.DFA, len(deserializedATN.DecisionToState))
	for index, ds := range deserializedATN.DecisionToState {
		decisionToDFA[index] = antlr.NewDFA(ds, index)
	}

	this.BaseParser = antlr.NewBaseParser(input)

	this.Interpreter = antlr.NewParserATNSimulator(this, deserializedATN, decisionToDFA, antlr.NewPredictionContextCache())
	this.RuleNames = ruleNames
	this.LiteralNames = literalNames
	this.SymbolicNames = symbolicNames
	this.GrammarFileName = "CEL.g4"

	return this
}

// CELParser tokens.
const (
	CELParserEOF            = antlr.TokenEOF
	CELParserEQUALS         = 1
	CELParserNOT_EQUALS     = 2
	CELParserIN             = 3
	CELParserLESS           = 4
	CELParserLESS_EQUALS    = 5
	CELParserGREATER_EQUALS = 6
	CELParserGREATER        = 7
	CELParserLOGICAL_AND    = 8
	CELParserLOGICAL_OR     = 9
	CELParserLBRACKET       = 10
	CELParserRPRACKET       = 11
	CELParserLBRACE         = 12
	CELParserRBRACE         = 13
	CELParserLPAREN         = 14
	CELParserRPAREN         = 15
	CELParserDOT            = 16
	CELParserCOMMA          = 17
	CELParserMINUS          = 18
	CELParserEXCLAM         = 19
	CELParserQUESTIONMARK   = 20
	CELParserCOLON          = 21
	CELParserPLUS           = 22
	CELParserSTAR           = 23
	CELParserSLASH          = 24
	CELParserPERCENT        = 25
	CELParserTRUE           = 26
	CELParserFALSE          = 27
	CELParserNULL           = 28
	CELParserWHITESPACE     = 29
	CELParserCOMMENT        = 30
	CELParserNUM_FLOAT      = 31
	CELParserNUM_INT        = 32
	CELParserNUM_UINT       = 33
	CELParserSTRING         = 34
	CELParserBYTES          = 35
	CELParserIDENTIFIER     = 36
)

// CELParser rules.
const (
	CELParserRULE_start                = 0
	CELParserRULE_expr                 = 1
	CELParserRULE_conditionalOr        = 2
	CELParserRULE_conditionalAnd       = 3
	CELParserRULE_relation             = 4
	CELParserRULE_calc                 = 5
	CELParserRULE_unary                = 6
	CELParserRULE_member               = 7
	CELParserRULE_primary              = 8
	CELParserRULE_exprList             = 9
	CELParserRULE_fieldInitializerList = 10
	CELParserRULE_mapInitializerList   = 11
	CELParserRULE_literal              = 12
)

// IStartContext is an interface to support dynamic dispatch.
type IStartContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// GetE returns the e rule contexts.
	GetE() IExprContext

	// SetE sets the e rule contexts.
	SetE(IExprContext)

	// IsStartContext differentiates from other interfaces.
	IsStartContext()
}

type StartContext struct {
	*antlr.BaseParserRuleContext
	parser antlr.Parser
	e      IExprContext
}

func NewEmptyStartContext() *StartContext {
	var p = new(StartContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_start
	return p
}

func (*StartContext) IsStartContext() {}

func NewStartContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *StartContext {
	var p = new(StartContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_start

	return p
}

func (s *StartContext) GetParser() antlr.Parser { return s.parser }

func (s *StartContext) GetE() IExprContext { return s.e }

func (s *StartContext) SetE(v IExprContext) { s.e = v }

func (s *StartContext) EOF() antlr.TerminalNode {
	return s.GetToken(CELParserEOF, 0)
}

func (s *StartContext) Expr() IExprContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IExprContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *StartContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *StartContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (s *StartContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterStart(s)
	}
}

func (s *StartContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitStart(s)
	}
}

func (s *StartContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitStart(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) Start() (localctx IStartContext) {
	localctx = NewStartContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 0, CELParserRULE_start)

	defer func() {
		p.ExitRule()
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(26)

		var _x = p.Expr()

		localctx.(*StartContext).e = _x
	}
	{
		p.SetState(27)
		p.Match(CELParserEOF)
	}

	return localctx
}

// IExprContext is an interface to support dynamic dispatch.
type IExprContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// GetOp returns the op token.
	GetOp() antlr.Token

	// SetOp sets the op token.
	SetOp(antlr.Token)

	// GetE returns the e rule contexts.
	GetE() IConditionalOrContext

	// GetE1 returns the e1 rule contexts.
	GetE1() IConditionalOrContext

	// GetE2 returns the e2 rule contexts.
	GetE2() IExprContext

	// SetE sets the e rule contexts.
	SetE(IConditionalOrContext)

	// SetE1 sets the e1 rule contexts.
	SetE1(IConditionalOrContext)

	// SetE2 sets the e2 rule contexts.
	SetE2(IExprContext)

	// IsExprContext differentiates from other interfaces.
	IsExprContext()
}

type ExprContext struct {
	*antlr.BaseParserRuleContext
	parser antlr.Parser
	e      IConditionalOrContext
	op     antlr.Token
	e1     IConditionalOrContext
	e2     IExprContext
}

func NewEmptyExprContext() *ExprContext {
	var p = new(ExprContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_expr
	return p
}

func (*ExprContext) IsExprContext() {}

func NewExprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *ExprContext {
	var p = new(ExprContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_expr

	return p
}

func (s *ExprContext) GetParser() antlr.Parser { return s.parser }

func (s *ExprContext) GetOp() antlr.Token { return s.op }

func (s *ExprContext) SetOp(v antlr.Token) { s.op = v }

func (s *ExprContext) GetE() IConditionalOrContext { return s.e }

func (s *ExprContext) GetE1() IConditionalOrContext { return s.e1 }

func (s *ExprContext) GetE2() IExprContext { return s.e2 }

func (s *ExprContext) SetE(v IConditionalOrContext) { s.e = v }

func (s *ExprContext) SetE1(v IConditionalOrContext) { s.e1 = v }

func (s *ExprContext) SetE2(v IExprContext) { s.e2 = v }

func (s *ExprContext) AllConditionalOr() []IConditionalOrContext {
	var ts = s.GetTypedRuleContexts(reflect.TypeOf((*IConditionalOrContext)(nil)).Elem())
	var tst = make([]IConditionalOrContext, len(ts))

	for i, t := range ts {
		if t != nil {
			tst[i] = t.(IConditionalOrContext)
		}
	}

	return tst
}

func (s *ExprContext) ConditionalOr(i int) IConditionalOrContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IConditionalOrContext)(nil)).Elem(), i)

	if t == nil {
		return nil
	}

	return t.(IConditionalOrContext)
}

func (s *ExprContext) Expr() IExprContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IExprContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *ExprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *ExprContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (s *ExprContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterExpr(s)
	}
}

func (s *ExprContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitExpr(s)
	}
}

func (s *ExprContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitExpr(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) Expr() (localctx IExprContext) {
	localctx = NewExprContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 2, CELParserRULE_expr)
	var _la int

	defer func() {
		p.ExitRule()
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(29)

		var _x = p.ConditionalOr()

		localctx.(*ExprContext).e = _x
	}
	p.SetState(35)
	p.GetErrorHandler().Sync(p)
	_la = p.GetTokenStream().LA(1)

	if _la == CELParserQUESTIONMARK {
		{
			p.SetState(30)

			var _m = p.Match(CELParserQUESTIONMARK)

			localctx.(*ExprContext).op = _m
		}
		{
			p.SetState(31)

			var _x = p.ConditionalOr()

			localctx.(*ExprContext).e1 = _x
		}
		{
			p.SetState(32)
			p.Match(CELParserCOLON)
		}
		{
			p.SetState(33)

			var _x = p.Expr()

			localctx.(*ExprContext).e2 = _x
		}

	}

	return localctx
}

// IConditionalOrContext is an interface to support dynamic dispatch.
type IConditionalOrContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// GetS9 returns the s9 token.
	GetS9() antlr.Token

	// SetS9 sets the s9 token.
	SetS9(antlr.Token)

	// GetOps returns the ops token list.
	GetOps() []antlr.Token

	// SetOps sets the ops token list.
	SetOps([]antlr.Token)

	// GetE returns the e rule contexts.
	GetE() IConditionalAndContext

	// Get_conditionalAnd returns the _conditionalAnd rule contexts.
	Get_conditionalAnd() IConditionalAndContext

	// SetE sets the e rule contexts.
	SetE(IConditionalAndContext)

	// Set_conditionalAnd sets the _conditionalAnd rule contexts.
	Set_conditionalAnd(IConditionalAndContext)

	// GetE1 returns the e1 rule context list.
	GetE1() []IConditionalAndContext

	// SetE1 sets the e1 rule context list.
	SetE1([]IConditionalAndContext)

	// IsConditionalOrContext differentiates from other interfaces.
	IsConditionalOrContext()
}

type ConditionalOrContext struct {
	*antlr.BaseParserRuleContext
	parser          antlr.Parser
	e               IConditionalAndContext
	s9              antlr.Token
	ops             []antlr.Token
	_conditionalAnd IConditionalAndContext
	e1              []IConditionalAndContext
}

func NewEmptyConditionalOrContext() *ConditionalOrContext {
	var p = new(ConditionalOrContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_conditionalOr
	return p
}

func (*ConditionalOrContext) IsConditionalOrContext() {}

func NewConditionalOrContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *ConditionalOrContext {
	var p = new(ConditionalOrContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_conditionalOr

	return p
}

func (s *ConditionalOrContext) GetParser() antlr.Parser { return s.parser }

func (s *ConditionalOrContext) GetS9() antlr.Token { return s.s9 }

func (s *ConditionalOrContext) SetS9(v antlr.Token) { s.s9 = v }

func (s *ConditionalOrContext) GetOps() []antlr.Token { return s.ops }

func (s *ConditionalOrContext) SetOps(v []antlr.Token) { s.ops = v }

func (s *ConditionalOrContext) GetE() IConditionalAndContext { return s.e }

func (s *ConditionalOrContext) Get_conditionalAnd() IConditionalAndContext { return s._conditionalAnd }

func (s *ConditionalOrContext) SetE(v IConditionalAndContext) { s.e = v }

func (s *ConditionalOrContext) Set_conditionalAnd(v IConditionalAndContext) { s._conditionalAnd = v }

func (s *ConditionalOrContext) GetE1() []IConditionalAndContext { return s.e1 }

func (s *ConditionalOrContext) SetE1(v []IConditionalAndContext) { s.e1 = v }

func (s *ConditionalOrContext) AllConditionalAnd() []IConditionalAndContext {
	var ts = s.GetTypedRuleContexts(reflect.TypeOf((*IConditionalAndContext)(nil)).Elem())
	var tst = make([]IConditionalAndContext, len(ts))

	for i, t := range ts {
		if t != nil {
			tst[i] = t.(IConditionalAndContext)
		}
	}

	return tst
}

func (s *ConditionalOrContext) ConditionalAnd(i int) IConditionalAndContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IConditionalAndContext)(nil)).Elem(), i)

	if t == nil {
		return nil
	}

	return t.(IConditionalAndContext)
}

func (s *ConditionalOrContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *ConditionalOrContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (s *ConditionalOrContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterConditionalOr(s)
	}
}

func (s *ConditionalOrContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitConditionalOr(s)
	}
}

func (s *ConditionalOrContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitConditionalOr(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) ConditionalOr() (localctx IConditionalOrContext) {
	localctx = NewConditionalOrContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 4, CELParserRULE_conditionalOr)
	var _la int

	defer func() {
		p.ExitRule()
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(37)

		var _x = p.ConditionalAnd()

		localctx.(*ConditionalOrContext).e = _x
	}
	p.SetState(42)
	p.GetErrorHandler().Sync(p)
	_la = p.GetTokenStream().LA(1)

	for _la == CELParserLOGICAL_OR {
		{
			p.SetState(38)

			var _m = p.Match(CELParserLOGICAL_OR)

			localctx.(*ConditionalOrContext).s9 = _m
		}
		localctx.(*ConditionalOrContext).ops = append(localctx.(*ConditionalOrContext).ops, localctx.(*ConditionalOrContext).s9)
		{
			p.SetState(39)

			var _x = p.ConditionalAnd()

			localctx.(*ConditionalOrContext)._conditionalAnd = _x
		}
		localctx.(*ConditionalOrContext).e1 = append(localctx.(*ConditionalOrContext).e1, localctx.(*ConditionalOrContext)._conditionalAnd)

		p.SetState(44)
		p.GetErrorHandler().Sync(p)
		_la = p.GetTokenStream().LA(1)
	}

	return localctx
}

// IConditionalAndContext is an interface to support dynamic dispatch.
type IConditionalAndContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// GetS8 returns the s8 token.
	GetS8() antlr.Token

	// SetS8 sets the s8 token.
	SetS8(antlr.Token)

	// GetOps returns the ops token list.
	GetOps() []antlr.Token

	// SetOps sets the ops token list.
	SetOps([]antlr.Token)

	// GetE returns the e rule contexts.
	GetE() IRelationContext

	// Get_relation returns the _relation rule contexts.
	Get_relation() IRelationContext

	// SetE sets the e rule contexts.
	SetE(IRelationContext)

	// Set_relation sets the _relation rule contexts.
	Set_relation(IRelationContext)

	// GetE1 returns the e1 rule context list.
	GetE1() []IRelationContext

	// SetE1 sets the e1 rule context list.
	SetE1([]IRelationContext)

	// IsConditionalAndContext differentiates from other interfaces.
	IsConditionalAndContext()
}

type ConditionalAndContext struct {
	*antlr.BaseParserRuleContext
	parser    antlr.Parser
	e         IRelationContext
	s8        antlr.Token
	ops       []antlr.Token
	_relation IRelationContext
	e1        []IRelationContext
}

func NewEmptyConditionalAndContext() *ConditionalAndContext {
	var p = new(ConditionalAndContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_conditionalAnd
	return p
}

func (*ConditionalAndContext) IsConditionalAndContext() {}

func NewConditionalAndContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *ConditionalAndContext {
	var p = new(ConditionalAndContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_conditionalAnd

	return p
}

func (s *ConditionalAndContext) GetParser() antlr.Parser { return s.parser }

func (s *ConditionalAndContext) GetS8() antlr.Token { return s.s8 }

func (s *ConditionalAndContext) SetS8(v antlr.Token) { s.s8 = v }

func (s *ConditionalAndContext) GetOps() []antlr.Token { return s.ops }

func (s *ConditionalAndContext) SetOps(v []antlr.Token) { s.ops = v }

func (s *ConditionalAndContext) GetE() IRelationContext { return s.e }

func (s *ConditionalAndContext) Get_relation() IRelationContext { return s._relation }

func (s *ConditionalAndContext) SetE(v IRelationContext) { s.e = v }

func (s *ConditionalAndContext) Set_relation(v IRelationContext) { s._relation = v }

func (s *ConditionalAndContext) GetE1() []IRelationContext { return s.e1 }

func (s *ConditionalAndContext) SetE1(v []IRelationContext) { s.e1 = v }

func (s *ConditionalAndContext) AllRelation() []IRelationContext {
	var ts = s.GetTypedRuleContexts(reflect.TypeOf((*IRelationContext)(nil)).Elem())
	var tst = make([]IRelationContext, len(ts))

	for i, t := range ts {
		if t != nil {
			tst[i] = t.(IRelationContext)
		}
	}

	return tst
}

func (s *ConditionalAndContext) Relation(i int) IRelationContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IRelationContext)(nil)).Elem(), i)

	if t == nil {
		return nil
	}

	return t.(IRelationContext)
}

func (s *ConditionalAndContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *ConditionalAndContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (s *ConditionalAndContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterConditionalAnd(s)
	}
}

func (s *ConditionalAndContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitConditionalAnd(s)
	}
}

func (s *ConditionalAndContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitConditionalAnd(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) ConditionalAnd() (localctx IConditionalAndContext) {
	localctx = NewConditionalAndContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 6, CELParserRULE_conditionalAnd)
	var _la int

	defer func() {
		p.ExitRule()
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(45)

		var _x = p.relation(0)

		localctx.(*ConditionalAndContext).e = _x
	}
	p.SetState(50)
	p.GetErrorHandler().Sync(p)
	_la = p.GetTokenStream().LA(1)

	for _la == CELParserLOGICAL_AND {
		{
			p.SetState(46)

			var _m = p.Match(CELParserLOGICAL_AND)

			localctx.(*ConditionalAndContext).s8 = _m
		}
		localctx.(*ConditionalAndContext).ops = append(localctx.(*ConditionalAndContext).ops, localctx.(*ConditionalAndContext).s8)
		{
			p.SetState(47)

			var _x = p.relation(0)

			localctx.(*ConditionalAndContext)._relation = _x
		}
		localctx.(*ConditionalAndContext).e1 = append(localctx.(*ConditionalAndContext).e1, localctx.(*ConditionalAndContext)._relation)

		p.SetState(52)
		p.GetErrorHandler().Sync(p)
		_la = p.GetTokenStream().LA(1)
	}

	return localctx
}

// IRelationContext is an interface to support dynamic dispatch.
type IRelationContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// GetOp returns the op token.
	GetOp() antlr.Token

	// SetOp sets the op token.
	SetOp(antlr.Token)

	// IsRelationContext differentiates from other interfaces.
	IsRelationContext()
}

type RelationContext struct {
	*antlr.BaseParserRuleContext
	parser antlr.Parser
	op     antlr.Token
}

func NewEmptyRelationContext() *RelationContext {
	var p = new(RelationContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_relation
	return p
}

func (*RelationContext) IsRelationContext() {}

func NewRelationContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *RelationContext {
	var p = new(RelationContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_relation

	return p
}

func (s *RelationContext) GetParser() antlr.Parser { return s.parser }

func (s *RelationContext) GetOp() antlr.Token { return s.op }

func (s *RelationContext) SetOp(v antlr.Token) { s.op = v }

func (s *RelationContext) Calc() ICalcContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*ICalcContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(ICalcContext)
}

func (s *RelationContext) AllRelation() []IRelationContext {
	var ts = s.GetTypedRuleContexts(reflect.TypeOf((*IRelationContext)(nil)).Elem())
	var tst = make([]IRelationContext, len(ts))

	for i, t := range ts {
		if t != nil {
			tst[i] = t.(IRelationContext)
		}
	}

	return tst
}

func (s *RelationContext) Relation(i int) IRelationContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IRelationContext)(nil)).Elem(), i)

	if t == nil {
		return nil
	}

	return t.(IRelationContext)
}

func (s *RelationContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *RelationContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (s *RelationContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterRelation(s)
	}
}

func (s *RelationContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitRelation(s)
	}
}

func (s *RelationContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitRelation(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) Relation() (localctx IRelationContext) {
	return p.relation(0)
}

func (p *CELParser) relation(_p int) (localctx IRelationContext) {
	var _parentctx antlr.ParserRuleContext = p.GetParserRuleContext()
	_parentState := p.GetState()
	localctx = NewRelationContext(p, p.GetParserRuleContext(), _parentState)
	var _prevctx IRelationContext = localctx
	var _ antlr.ParserRuleContext = _prevctx // TODO: To prevent unused variable warning.
	_startState := 8
	p.EnterRecursionRule(localctx, 8, CELParserRULE_relation, _p)
	var _la int

	defer func() {
		p.UnrollRecursionContexts(_parentctx)
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	var _alt int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(54)
		p.calc(0)
	}

	p.GetParserRuleContext().SetStop(p.GetTokenStream().LT(-1))
	p.SetState(61)
	p.GetErrorHandler().Sync(p)
	_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 3, p.GetParserRuleContext())

	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			if p.GetParseListeners() != nil {
				p.TriggerExitRuleEvent()
			}
			_prevctx = localctx
			localctx = NewRelationContext(p, _parentctx, _parentState)
			p.PushNewRecursionContext(localctx, _startState, CELParserRULE_relation)
			p.SetState(56)

			if !(p.Precpred(p.GetParserRuleContext(), 1)) {
				panic(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 1)", ""))
			}
			p.SetState(57)

			var _lt = p.GetTokenStream().LT(1)

			localctx.(*RelationContext).op = _lt

			_la = p.GetTokenStream().LA(1)

			if !(((_la)&-(0x1f+1)) == 0 && ((1<<uint(_la))&((1<<CELParserEQUALS)|(1<<CELParserNOT_EQUALS)|(1<<CELParserIN)|(1<<CELParserLESS)|(1<<CELParserLESS_EQUALS)|(1<<CELParserGREATER_EQUALS)|(1<<CELParserGREATER))) != 0) {
				var _ri = p.GetErrorHandler().RecoverInline(p)

				localctx.(*RelationContext).op = _ri
			} else {
				p.GetErrorHandler().ReportMatch(p)
				p.Consume()
			}
			{
				p.SetState(58)
				p.relation(2)
			}

		}
		p.SetState(63)
		p.GetErrorHandler().Sync(p)
		_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 3, p.GetParserRuleContext())
	}

	return localctx
}

// ICalcContext is an interface to support dynamic dispatch.
type ICalcContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// GetOp returns the op token.
	GetOp() antlr.Token

	// SetOp sets the op token.
	SetOp(antlr.Token)

	// IsCalcContext differentiates from other interfaces.
	IsCalcContext()
}

type CalcContext struct {
	*antlr.BaseParserRuleContext
	parser antlr.Parser
	op     antlr.Token
}

func NewEmptyCalcContext() *CalcContext {
	var p = new(CalcContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_calc
	return p
}

func (*CalcContext) IsCalcContext() {}

func NewCalcContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *CalcContext {
	var p = new(CalcContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_calc

	return p
}

func (s *CalcContext) GetParser() antlr.Parser { return s.parser }

func (s *CalcContext) GetOp() antlr.Token { return s.op }

func (s *CalcContext) SetOp(v antlr.Token) { s.op = v }

func (s *CalcContext) Unary() IUnaryContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IUnaryContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IUnaryContext)
}

func (s *CalcContext) AllCalc() []ICalcContext {
	var ts = s.GetTypedRuleContexts(reflect.TypeOf((*ICalcContext)(nil)).Elem())
	var tst = make([]ICalcContext, len(ts))

	for i, t := range ts {
		if t != nil {
			tst[i] = t.(ICalcContext)
		}
	}

	return tst
}

func (s *CalcContext) Calc(i int) ICalcContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*ICalcContext)(nil)).Elem(), i)

	if t == nil {
		return nil
	}

	return t.(ICalcContext)
}

func (s *CalcContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *CalcContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (s *CalcContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterCalc(s)
	}
}

func (s *CalcContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitCalc(s)
	}
}

func (s *CalcContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitCalc(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) Calc() (localctx ICalcContext) {
	return p.calc(0)
}

func (p *CELParser) calc(_p int) (localctx ICalcContext) {
	var _parentctx antlr.ParserRuleContext = p.GetParserRuleContext()
	_parentState := p.GetState()
	localctx = NewCalcContext(p, p.GetParserRuleContext(), _parentState)
	var _prevctx ICalcContext = localctx
	var _ antlr.ParserRuleContext = _prevctx // TODO: To prevent unused variable warning.
	_startState := 10
	p.EnterRecursionRule(localctx, 10, CELParserRULE_calc, _p)
	var _la int

	defer func() {
		p.UnrollRecursionContexts(_parentctx)
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	var _alt int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(65)
		p.Unary()
	}

	p.GetParserRuleContext().SetStop(p.GetTokenStream().LT(-1))
	p.SetState(75)
	p.GetErrorHandler().Sync(p)
	_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 5, p.GetParserRuleContext())

	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			if p.GetParseListeners() != nil {
				p.TriggerExitRuleEvent()
			}
			_prevctx = localctx
			p.SetState(73)
			p.GetErrorHandler().Sync(p)
			switch p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 4, p.GetParserRuleContext()) {
			case 1:
				localctx = NewCalcContext(p, _parentctx, _parentState)
				p.PushNewRecursionContext(localctx, _startState, CELParserRULE_calc)
				p.SetState(67)

				if !(p.Precpred(p.GetParserRuleContext(), 2)) {
					panic(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 2)", ""))
				}
				p.SetState(68)

				var _lt = p.GetTokenStream().LT(1)

				localctx.(*CalcContext).op = _lt

				_la = p.GetTokenStream().LA(1)

				if !(((_la)&-(0x1f+1)) == 0 && ((1<<uint(_la))&((1<<CELParserSTAR)|(1<<CELParserSLASH)|(1<<CELParserPERCENT))) != 0) {
					var _ri = p.GetErrorHandler().RecoverInline(p)

					localctx.(*CalcContext).op = _ri
				} else {
					p.GetErrorHandler().ReportMatch(p)
					p.Consume()
				}
				{
					p.SetState(69)
					p.calc(3)
				}

			case 2:
				localctx = NewCalcContext(p, _parentctx, _parentState)
				p.PushNewRecursionContext(localctx, _startState, CELParserRULE_calc)
				p.SetState(70)

				if !(p.Precpred(p.GetParserRuleContext(), 1)) {
					panic(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 1)", ""))
				}
				p.SetState(71)

				var _lt = p.GetTokenStream().LT(1)

				localctx.(*CalcContext).op = _lt

				_la = p.GetTokenStream().LA(1)

				if !(_la == CELParserMINUS || _la == CELParserPLUS) {
					var _ri = p.GetErrorHandler().RecoverInline(p)

					localctx.(*CalcContext).op = _ri
				} else {
					p.GetErrorHandler().ReportMatch(p)
					p.Consume()
				}
				{
					p.SetState(72)
					p.calc(2)
				}

			}

		}
		p.SetState(77)
		p.GetErrorHandler().Sync(p)
		_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 5, p.GetParserRuleContext())
	}

	return localctx
}

// IUnaryContext is an interface to support dynamic dispatch.
type IUnaryContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// IsUnaryContext differentiates from other interfaces.
	IsUnaryContext()
}

type UnaryContext struct {
	*antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyUnaryContext() *UnaryContext {
	var p = new(UnaryContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_unary
	return p
}

func (*UnaryContext) IsUnaryContext() {}

func NewUnaryContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *UnaryContext {
	var p = new(UnaryContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_unary

	return p
}

func (s *UnaryContext) GetParser() antlr.Parser { return s.parser }

func (s *UnaryContext) CopyFrom(ctx *UnaryContext) {
	s.BaseParserRuleContext.CopyFrom(ctx.BaseParserRuleContext)
}

func (s *UnaryContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *UnaryContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

type LogicalNotContext struct {
	*UnaryContext
	s19 antlr.Token
	ops []antlr.Token
}

func NewLogicalNotContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *LogicalNotContext {
	var p = new(LogicalNotContext)

	p.UnaryContext = NewEmptyUnaryContext()
	p.parser = parser
	p.CopyFrom(ctx.(*UnaryContext))

	return p
}

func (s *LogicalNotContext) GetS19() antlr.Token { return s.s19 }

func (s *LogicalNotContext) SetS19(v antlr.Token) { s.s19 = v }

func (s *LogicalNotContext) GetOps() []antlr.Token { return s.ops }

func (s *LogicalNotContext) SetOps(v []antlr.Token) { s.ops = v }

func (s *LogicalNotContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *LogicalNotContext) Member() IMemberContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IMemberContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IMemberContext)
}

func (s *LogicalNotContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterLogicalNot(s)
	}
}

func (s *LogicalNotContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitLogicalNot(s)
	}
}

func (s *LogicalNotContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitLogicalNot(s)

	default:
		return t.VisitChildren(s)
	}
}

type MemberExprContext struct {
	*UnaryContext
}

func NewMemberExprContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *MemberExprContext {
	var p = new(MemberExprContext)

	p.UnaryContext = NewEmptyUnaryContext()
	p.parser = parser
	p.CopyFrom(ctx.(*UnaryContext))

	return p
}

func (s *MemberExprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *MemberExprContext) Member() IMemberContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IMemberContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IMemberContext)
}

func (s *MemberExprContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterMemberExpr(s)
	}
}

func (s *MemberExprContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitMemberExpr(s)
	}
}

func (s *MemberExprContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitMemberExpr(s)

	default:
		return t.VisitChildren(s)
	}
}

type NegateContext struct {
	*UnaryContext
	s18 antlr.Token
	ops []antlr.Token
}

func NewNegateContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *NegateContext {
	var p = new(NegateContext)

	p.UnaryContext = NewEmptyUnaryContext()
	p.parser = parser
	p.CopyFrom(ctx.(*UnaryContext))

	return p
}

func (s *NegateContext) GetS18() antlr.Token { return s.s18 }

func (s *NegateContext) SetS18(v antlr.Token) { s.s18 = v }

func (s *NegateContext) GetOps() []antlr.Token { return s.ops }

func (s *NegateContext) SetOps(v []antlr.Token) { s.ops = v }

func (s *NegateContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *NegateContext) Member() IMemberContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IMemberContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IMemberContext)
}

func (s *NegateContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterNegate(s)
	}
}

func (s *NegateContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitNegate(s)
	}
}

func (s *NegateContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitNegate(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) Unary() (localctx IUnaryContext) {
	localctx = NewUnaryContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 12, CELParserRULE_unary)
	var _la int

	defer func() {
		p.ExitRule()
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	var _alt int

	p.SetState(91)
	p.GetErrorHandler().Sync(p)
	switch p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 8, p.GetParserRuleContext()) {
	case 1:
		localctx = NewMemberExprContext(p, localctx)
		p.EnterOuterAlt(localctx, 1)
		{
			p.SetState(78)
			p.member(0)
		}

	case 2:
		localctx = NewLogicalNotContext(p, localctx)
		p.EnterOuterAlt(localctx, 2)
		p.SetState(80)
		p.GetErrorHandler().Sync(p)
		_la = p.GetTokenStream().LA(1)

		for ok := true; ok; ok = _la == CELParserEXCLAM {
			{
				p.SetState(79)

				var _m = p.Match(CELParserEXCLAM)

				localctx.(*LogicalNotContext).s19 = _m
			}
			localctx.(*LogicalNotContext).ops = append(localctx.(*LogicalNotContext).ops, localctx.(*LogicalNotContext).s19)

			p.SetState(82)
			p.GetErrorHandler().Sync(p)
			_la = p.GetTokenStream().LA(1)
		}
		{
			p.SetState(84)
			p.member(0)
		}

	case 3:
		localctx = NewNegateContext(p, localctx)
		p.EnterOuterAlt(localctx, 3)
		p.SetState(86)
		p.GetErrorHandler().Sync(p)
		_alt = 1
		for ok := true; ok; ok = _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
			switch _alt {
			case 1:
				{
					p.SetState(85)

					var _m = p.Match(CELParserMINUS)

					localctx.(*NegateContext).s18 = _m
				}
				localctx.(*NegateContext).ops = append(localctx.(*NegateContext).ops, localctx.(*NegateContext).s18)

			default:
				panic(antlr.NewNoViableAltException(p, nil, nil, nil, nil, nil))
			}

			p.SetState(88)
			p.GetErrorHandler().Sync(p)
			_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 7, p.GetParserRuleContext())
		}
		{
			p.SetState(90)
			p.member(0)
		}

	}

	return localctx
}

// IMemberContext is an interface to support dynamic dispatch.
type IMemberContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// IsMemberContext differentiates from other interfaces.
	IsMemberContext()
}

type MemberContext struct {
	*antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyMemberContext() *MemberContext {
	var p = new(MemberContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_member
	return p
}

func (*MemberContext) IsMemberContext() {}

func NewMemberContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *MemberContext {
	var p = new(MemberContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_member

	return p
}

func (s *MemberContext) GetParser() antlr.Parser { return s.parser }

func (s *MemberContext) CopyFrom(ctx *MemberContext) {
	s.BaseParserRuleContext.CopyFrom(ctx.BaseParserRuleContext)
}

func (s *MemberContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *MemberContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

type SelectOrCallContext struct {
	*MemberContext
	op   antlr.Token
	id   antlr.Token
	open antlr.Token
	args IExprListContext
}

func NewSelectOrCallContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *SelectOrCallContext {
	var p = new(SelectOrCallContext)

	p.MemberContext = NewEmptyMemberContext()
	p.parser = parser
	p.CopyFrom(ctx.(*MemberContext))

	return p
}

func (s *SelectOrCallContext) GetOp() antlr.Token { return s.op }

func (s *SelectOrCallContext) GetId() antlr.Token { return s.id }

func (s *SelectOrCallContext) GetOpen() antlr.Token { return s.open }

func (s *SelectOrCallContext) SetOp(v antlr.Token) { s.op = v }

func (s *SelectOrCallContext) SetId(v antlr.Token) { s.id = v }

func (s *SelectOrCallContext) SetOpen(v antlr.Token) { s.open = v }

func (s *SelectOrCallContext) GetArgs() IExprListContext { return s.args }

func (s *SelectOrCallContext) SetArgs(v IExprListContext) { s.args = v }

func (s *SelectOrCallContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *SelectOrCallContext) Member() IMemberContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IMemberContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IMemberContext)
}

func (s *SelectOrCallContext) IDENTIFIER() antlr.TerminalNode {
	return s.GetToken(CELParserIDENTIFIER, 0)
}

func (s *SelectOrCallContext) ExprList() IExprListContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IExprListContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IExprListContext)
}

func (s *SelectOrCallContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterSelectOrCall(s)
	}
}

func (s *SelectOrCallContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitSelectOrCall(s)
	}
}

func (s *SelectOrCallContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitSelectOrCall(s)

	default:
		return t.VisitChildren(s)
	}
}

type PrimaryExprContext struct {
	*MemberContext
}

func NewPrimaryExprContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *PrimaryExprContext {
	var p = new(PrimaryExprContext)

	p.MemberContext = NewEmptyMemberContext()
	p.parser = parser
	p.CopyFrom(ctx.(*MemberContext))

	return p
}

func (s *PrimaryExprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *PrimaryExprContext) Primary() IPrimaryContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IPrimaryContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IPrimaryContext)
}

func (s *PrimaryExprContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterPrimaryExpr(s)
	}
}

func (s *PrimaryExprContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitPrimaryExpr(s)
	}
}

func (s *PrimaryExprContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitPrimaryExpr(s)

	default:
		return t.VisitChildren(s)
	}
}

type IndexContext struct {
	*MemberContext
	op    antlr.Token
	index IExprContext
}

func NewIndexContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *IndexContext {
	var p = new(IndexContext)

	p.MemberContext = NewEmptyMemberContext()
	p.parser = parser
	p.CopyFrom(ctx.(*MemberContext))

	return p
}

func (s *IndexContext) GetOp() antlr.Token { return s.op }

func (s *IndexContext) SetOp(v antlr.Token) { s.op = v }

func (s *IndexContext) GetIndex() IExprContext { return s.index }

func (s *IndexContext) SetIndex(v IExprContext) { s.index = v }

func (s *IndexContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *IndexContext) Member() IMemberContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IMemberContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IMemberContext)
}

func (s *IndexContext) Expr() IExprContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IExprContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *IndexContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterIndex(s)
	}
}

func (s *IndexContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitIndex(s)
	}
}

func (s *IndexContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitIndex(s)

	default:
		return t.VisitChildren(s)
	}
}

type CreateMessageContext struct {
	*MemberContext
	op      antlr.Token
	entries IFieldInitializerListContext
}

func NewCreateMessageContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *CreateMessageContext {
	var p = new(CreateMessageContext)

	p.MemberContext = NewEmptyMemberContext()
	p.parser = parser
	p.CopyFrom(ctx.(*MemberContext))

	return p
}

func (s *CreateMessageContext) GetOp() antlr.Token { return s.op }

func (s *CreateMessageContext) SetOp(v antlr.Token) { s.op = v }

func (s *CreateMessageContext) GetEntries() IFieldInitializerListContext { return s.entries }

func (s *CreateMessageContext) SetEntries(v IFieldInitializerListContext) { s.entries = v }

func (s *CreateMessageContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *CreateMessageContext) Member() IMemberContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IMemberContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IMemberContext)
}

func (s *CreateMessageContext) FieldInitializerList() IFieldInitializerListContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IFieldInitializerListContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IFieldInitializerListContext)
}

func (s *CreateMessageContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterCreateMessage(s)
	}
}

func (s *CreateMessageContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitCreateMessage(s)
	}
}

func (s *CreateMessageContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitCreateMessage(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) Member() (localctx IMemberContext) {
	return p.member(0)
}

func (p *CELParser) member(_p int) (localctx IMemberContext) {
	var _parentctx antlr.ParserRuleContext = p.GetParserRuleContext()
	_parentState := p.GetState()
	localctx = NewMemberContext(p, p.GetParserRuleContext(), _parentState)
	var _prevctx IMemberContext = localctx
	var _ antlr.ParserRuleContext = _prevctx // TODO: To prevent unused variable warning.
	_startState := 14
	p.EnterRecursionRule(localctx, 14, CELParserRULE_member, _p)
	var _la int

	defer func() {
		p.UnrollRecursionContexts(_parentctx)
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	var _alt int

	p.EnterOuterAlt(localctx, 1)
	localctx = NewPrimaryExprContext(p, localctx)
	p.SetParserRuleContext(localctx)
	_prevctx = localctx

	{
		p.SetState(94)
		p.Primary()
	}

	p.GetParserRuleContext().SetStop(p.GetTokenStream().LT(-1))
	p.SetState(122)
	p.GetErrorHandler().Sync(p)
	_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 14, p.GetParserRuleContext())

	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			if p.GetParseListeners() != nil {
				p.TriggerExitRuleEvent()
			}
			_prevctx = localctx
			p.SetState(120)
			p.GetErrorHandler().Sync(p)
			switch p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 13, p.GetParserRuleContext()) {
			case 1:
				localctx = NewSelectOrCallContext(p, NewMemberContext(p, _parentctx, _parentState))
				p.PushNewRecursionContext(localctx, _startState, CELParserRULE_member)
				p.SetState(96)

				if !(p.Precpred(p.GetParserRuleContext(), 3)) {
					panic(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 3)", ""))
				}
				{
					p.SetState(97)

					var _m = p.Match(CELParserDOT)

					localctx.(*SelectOrCallContext).op = _m
				}
				{
					p.SetState(98)

					var _m = p.Match(CELParserIDENTIFIER)

					localctx.(*SelectOrCallContext).id = _m
				}
				p.SetState(104)
				p.GetErrorHandler().Sync(p)

				if p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 10, p.GetParserRuleContext()) == 1 {
					{
						p.SetState(99)

						var _m = p.Match(CELParserLPAREN)

						localctx.(*SelectOrCallContext).open = _m
					}
					p.SetState(101)
					p.GetErrorHandler().Sync(p)
					_la = p.GetTokenStream().LA(1)

					if ((_la-10)&-(0x1f+1)) == 0 && ((1<<uint((_la-10)))&((1<<(CELParserLBRACKET-10))|(1<<(CELParserLBRACE-10))|(1<<(CELParserLPAREN-10))|(1<<(CELParserDOT-10))|(1<<(CELParserMINUS-10))|(1<<(CELParserEXCLAM-10))|(1<<(CELParserTRUE-10))|(1<<(CELParserFALSE-10))|(1<<(CELParserNULL-10))|(1<<(CELParserNUM_FLOAT-10))|(1<<(CELParserNUM_INT-10))|(1<<(CELParserNUM_UINT-10))|(1<<(CELParserSTRING-10))|(1<<(CELParserBYTES-10))|(1<<(CELParserIDENTIFIER-10)))) != 0 {
						{
							p.SetState(100)

							var _x = p.ExprList()

							localctx.(*SelectOrCallContext).args = _x
						}

					}
					{
						p.SetState(103)
						p.Match(CELParserRPAREN)
					}

				}

			case 2:
				localctx = NewIndexContext(p, NewMemberContext(p, _parentctx, _parentState))
				p.PushNewRecursionContext(localctx, _startState, CELParserRULE_member)
				p.SetState(106)

				if !(p.Precpred(p.GetParserRuleContext(), 2)) {
					panic(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 2)", ""))
				}
				{
					p.SetState(107)

					var _m = p.Match(CELParserLBRACKET)

					localctx.(*IndexContext).op = _m
				}
				{
					p.SetState(108)

					var _x = p.Expr()

					localctx.(*IndexContext).index = _x
				}
				{
					p.SetState(109)
					p.Match(CELParserRPRACKET)
				}

			case 3:
				localctx = NewCreateMessageContext(p, NewMemberContext(p, _parentctx, _parentState))
				p.PushNewRecursionContext(localctx, _startState, CELParserRULE_member)
				p.SetState(111)

				if !(p.Precpred(p.GetParserRuleContext(), 1)) {
					panic(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 1)", ""))
				}
				{
					p.SetState(112)

					var _m = p.Match(CELParserLBRACE)

					localctx.(*CreateMessageContext).op = _m
				}
				p.SetState(114)
				p.GetErrorHandler().Sync(p)
				_la = p.GetTokenStream().LA(1)

				if _la == CELParserIDENTIFIER {
					{
						p.SetState(113)

						var _x = p.FieldInitializerList()

						localctx.(*CreateMessageContext).entries = _x
					}

				}
				p.SetState(117)
				p.GetErrorHandler().Sync(p)
				_la = p.GetTokenStream().LA(1)

				if _la == CELParserCOMMA {
					{
						p.SetState(116)
						p.Match(CELParserCOMMA)
					}

				}
				{
					p.SetState(119)
					p.Match(CELParserRBRACE)
				}

			}

		}
		p.SetState(124)
		p.GetErrorHandler().Sync(p)
		_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 14, p.GetParserRuleContext())
	}

	return localctx
}

// IPrimaryContext is an interface to support dynamic dispatch.
type IPrimaryContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// IsPrimaryContext differentiates from other interfaces.
	IsPrimaryContext()
}

type PrimaryContext struct {
	*antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyPrimaryContext() *PrimaryContext {
	var p = new(PrimaryContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_primary
	return p
}

func (*PrimaryContext) IsPrimaryContext() {}

func NewPrimaryContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *PrimaryContext {
	var p = new(PrimaryContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_primary

	return p
}

func (s *PrimaryContext) GetParser() antlr.Parser { return s.parser }

func (s *PrimaryContext) CopyFrom(ctx *PrimaryContext) {
	s.BaseParserRuleContext.CopyFrom(ctx.BaseParserRuleContext)
}

func (s *PrimaryContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *PrimaryContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

type CreateListContext struct {
	*PrimaryContext
	op    antlr.Token
	elems IExprListContext
}

func NewCreateListContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *CreateListContext {
	var p = new(CreateListContext)

	p.PrimaryContext = NewEmptyPrimaryContext()
	p.parser = parser
	p.CopyFrom(ctx.(*PrimaryContext))

	return p
}

func (s *CreateListContext) GetOp() antlr.Token { return s.op }

func (s *CreateListContext) SetOp(v antlr.Token) { s.op = v }

func (s *CreateListContext) GetElems() IExprListContext { return s.elems }

func (s *CreateListContext) SetElems(v IExprListContext) { s.elems = v }

func (s *CreateListContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *CreateListContext) ExprList() IExprListContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IExprListContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IExprListContext)
}

func (s *CreateListContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterCreateList(s)
	}
}

func (s *CreateListContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitCreateList(s)
	}
}

func (s *CreateListContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitCreateList(s)

	default:
		return t.VisitChildren(s)
	}
}

type CreateStructContext struct {
	*PrimaryContext
	op      antlr.Token
	entries IMapInitializerListContext
}

func NewCreateStructContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *CreateStructContext {
	var p = new(CreateStructContext)

	p.PrimaryContext = NewEmptyPrimaryContext()
	p.parser = parser
	p.CopyFrom(ctx.(*PrimaryContext))

	return p
}

func (s *CreateStructContext) GetOp() antlr.Token { return s.op }

func (s *CreateStructContext) SetOp(v antlr.Token) { s.op = v }

func (s *CreateStructContext) GetEntries() IMapInitializerListContext { return s.entries }

func (s *CreateStructContext) SetEntries(v IMapInitializerListContext) { s.entries = v }

func (s *CreateStructContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *CreateStructContext) MapInitializerList() IMapInitializerListContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IMapInitializerListContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IMapInitializerListContext)
}

func (s *CreateStructContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterCreateStruct(s)
	}
}

func (s *CreateStructContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitCreateStruct(s)
	}
}

func (s *CreateStructContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitCreateStruct(s)

	default:
		return t.VisitChildren(s)
	}
}

type ConstantLiteralContext struct {
	*PrimaryContext
}

func NewConstantLiteralContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *ConstantLiteralContext {
	var p = new(ConstantLiteralContext)

	p.PrimaryContext = NewEmptyPrimaryContext()
	p.parser = parser
	p.CopyFrom(ctx.(*PrimaryContext))

	return p
}

func (s *ConstantLiteralContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *ConstantLiteralContext) Literal() ILiteralContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*ILiteralContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(ILiteralContext)
}

func (s *ConstantLiteralContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterConstantLiteral(s)
	}
}

func (s *ConstantLiteralContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitConstantLiteral(s)
	}
}

func (s *ConstantLiteralContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitConstantLiteral(s)

	default:
		return t.VisitChildren(s)
	}
}

type NestedContext struct {
	*PrimaryContext
	e IExprContext
}

func NewNestedContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *NestedContext {
	var p = new(NestedContext)

	p.PrimaryContext = NewEmptyPrimaryContext()
	p.parser = parser
	p.CopyFrom(ctx.(*PrimaryContext))

	return p
}

func (s *NestedContext) GetE() IExprContext { return s.e }

func (s *NestedContext) SetE(v IExprContext) { s.e = v }

func (s *NestedContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *NestedContext) Expr() IExprContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IExprContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *NestedContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterNested(s)
	}
}

func (s *NestedContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitNested(s)
	}
}

func (s *NestedContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitNested(s)

	default:
		return t.VisitChildren(s)
	}
}

type IdentOrGlobalCallContext struct {
	*PrimaryContext
	leadingDot antlr.Token
	id         antlr.Token
	op         antlr.Token
	args       IExprListContext
}

func NewIdentOrGlobalCallContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *IdentOrGlobalCallContext {
	var p = new(IdentOrGlobalCallContext)

	p.PrimaryContext = NewEmptyPrimaryContext()
	p.parser = parser
	p.CopyFrom(ctx.(*PrimaryContext))

	return p
}

func (s *IdentOrGlobalCallContext) GetLeadingDot() antlr.Token { return s.leadingDot }

func (s *IdentOrGlobalCallContext) GetId() antlr.Token { return s.id }

func (s *IdentOrGlobalCallContext) GetOp() antlr.Token { return s.op }

func (s *IdentOrGlobalCallContext) SetLeadingDot(v antlr.Token) { s.leadingDot = v }

func (s *IdentOrGlobalCallContext) SetId(v antlr.Token) { s.id = v }

func (s *IdentOrGlobalCallContext) SetOp(v antlr.Token) { s.op = v }

func (s *IdentOrGlobalCallContext) GetArgs() IExprListContext { return s.args }

func (s *IdentOrGlobalCallContext) SetArgs(v IExprListContext) { s.args = v }

func (s *IdentOrGlobalCallContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *IdentOrGlobalCallContext) IDENTIFIER() antlr.TerminalNode {
	return s.GetToken(CELParserIDENTIFIER, 0)
}

func (s *IdentOrGlobalCallContext) ExprList() IExprListContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IExprListContext)(nil)).Elem(), 0)

	if t == nil {
		return nil
	}

	return t.(IExprListContext)
}

func (s *IdentOrGlobalCallContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterIdentOrGlobalCall(s)
	}
}

func (s *IdentOrGlobalCallContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitIdentOrGlobalCall(s)
	}
}

func (s *IdentOrGlobalCallContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitIdentOrGlobalCall(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) Primary() (localctx IPrimaryContext) {
	localctx = NewPrimaryContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 16, CELParserRULE_primary)
	var _la int

	defer func() {
		p.ExitRule()
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	p.SetState(157)
	p.GetErrorHandler().Sync(p)

	switch p.GetTokenStream().LA(1) {
	case CELParserDOT, CELParserIDENTIFIER:
		localctx = NewIdentOrGlobalCallContext(p, localctx)
		p.EnterOuterAlt(localctx, 1)
		p.SetState(126)
		p.GetErrorHandler().Sync(p)
		_la = p.GetTokenStream().LA(1)

		if _la == CELParserDOT {
			{
				p.SetState(125)

				var _m = p.Match(CELParserDOT)

				localctx.(*IdentOrGlobalCallContext).leadingDot = _m
			}

		}
		{
			p.SetState(128)

			var _m = p.Match(CELParserIDENTIFIER)

			localctx.(*IdentOrGlobalCallContext).id = _m
		}
		p.SetState(134)
		p.GetErrorHandler().Sync(p)

		if p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 17, p.GetParserRuleContext()) == 1 {
			{
				p.SetState(129)

				var _m = p.Match(CELParserLPAREN)

				localctx.(*IdentOrGlobalCallContext).op = _m
			}
			p.SetState(131)
			p.GetErrorHandler().Sync(p)
			_la = p.GetTokenStream().LA(1)

			if ((_la-10)&-(0x1f+1)) == 0 && ((1<<uint((_la-10)))&((1<<(CELParserLBRACKET-10))|(1<<(CELParserLBRACE-10))|(1<<(CELParserLPAREN-10))|(1<<(CELParserDOT-10))|(1<<(CELParserMINUS-10))|(1<<(CELParserEXCLAM-10))|(1<<(CELParserTRUE-10))|(1<<(CELParserFALSE-10))|(1<<(CELParserNULL-10))|(1<<(CELParserNUM_FLOAT-10))|(1<<(CELParserNUM_INT-10))|(1<<(CELParserNUM_UINT-10))|(1<<(CELParserSTRING-10))|(1<<(CELParserBYTES-10))|(1<<(CELParserIDENTIFIER-10)))) != 0 {
				{
					p.SetState(130)

					var _x = p.ExprList()

					localctx.(*IdentOrGlobalCallContext).args = _x
				}

			}
			{
				p.SetState(133)
				p.Match(CELParserRPAREN)
			}

		}

	case CELParserLPAREN:
		localctx = NewNestedContext(p, localctx)
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(136)
			p.Match(CELParserLPAREN)
		}
		{
			p.SetState(137)

			var _x = p.Expr()

			localctx.(*NestedContext).e = _x
		}
		{
			p.SetState(138)
			p.Match(CELParserRPAREN)
		}

	case CELParserLBRACKET:
		localctx = NewCreateListContext(p, localctx)
		p.EnterOuterAlt(localctx, 3)
		{
			p.SetState(140)

			var _m = p.Match(CELParserLBRACKET)

			localctx.(*CreateListContext).op = _m
		}
		p.SetState(142)
		p.GetErrorHandler().Sync(p)
		_la = p.GetTokenStream().LA(1)

		if ((_la-10)&-(0x1f+1)) == 0 && ((1<<uint((_la-10)))&((1<<(CELParserLBRACKET-10))|(1<<(CELParserLBRACE-10))|(1<<(CELParserLPAREN-10))|(1<<(CELParserDOT-10))|(1<<(CELParserMINUS-10))|(1<<(CELParserEXCLAM-10))|(1<<(CELParserTRUE-10))|(1<<(CELParserFALSE-10))|(1<<(CELParserNULL-10))|(1<<(CELParserNUM_FLOAT-10))|(1<<(CELParserNUM_INT-10))|(1<<(CELParserNUM_UINT-10))|(1<<(CELParserSTRING-10))|(1<<(CELParserBYTES-10))|(1<<(CELParserIDENTIFIER-10)))) != 0 {
			{
				p.SetState(141)

				var _x = p.ExprList()

				localctx.(*CreateListContext).elems = _x
			}

		}
		p.SetState(145)
		p.GetErrorHandler().Sync(p)
		_la = p.GetTokenStream().LA(1)

		if _la == CELParserCOMMA {
			{
				p.SetState(144)
				p.Match(CELParserCOMMA)
			}

		}
		{
			p.SetState(147)
			p.Match(CELParserRPRACKET)
		}

	case CELParserLBRACE:
		localctx = NewCreateStructContext(p, localctx)
		p.EnterOuterAlt(localctx, 4)
		{
			p.SetState(148)

			var _m = p.Match(CELParserLBRACE)

			localctx.(*CreateStructContext).op = _m
		}
		p.SetState(150)
		p.GetErrorHandler().Sync(p)
		_la = p.GetTokenStream().LA(1)

		if ((_la-10)&-(0x1f+1)) == 0 && ((1<<uint((_la-10)))&((1<<(CELParserLBRACKET-10))|(1<<(CELParserLBRACE-10))|(1<<(CELParserLPAREN-10))|(1<<(CELParserDOT-10))|(1<<(CELParserMINUS-10))|(1<<(CELParserEXCLAM-10))|(1<<(CELParserTRUE-10))|(1<<(CELParserFALSE-10))|(1<<(CELParserNULL-10))|(1<<(CELParserNUM_FLOAT-10))|(1<<(CELParserNUM_INT-10))|(1<<(CELParserNUM_UINT-10))|(1<<(CELParserSTRING-10))|(1<<(CELParserBYTES-10))|(1<<(CELParserIDENTIFIER-10)))) != 0 {
			{
				p.SetState(149)

				var _x = p.MapInitializerList()

				localctx.(*CreateStructContext).entries = _x
			}

		}
		p.SetState(153)
		p.GetErrorHandler().Sync(p)
		_la = p.GetTokenStream().LA(1)

		if _la == CELParserCOMMA {
			{
				p.SetState(152)
				p.Match(CELParserCOMMA)
			}

		}
		{
			p.SetState(155)
			p.Match(CELParserRBRACE)
		}

	case CELParserMINUS, CELParserTRUE, CELParserFALSE, CELParserNULL, CELParserNUM_FLOAT, CELParserNUM_INT, CELParserNUM_UINT, CELParserSTRING, CELParserBYTES:
		localctx = NewConstantLiteralContext(p, localctx)
		p.EnterOuterAlt(localctx, 5)
		{
			p.SetState(156)
			p.Literal()
		}

	default:
		panic(antlr.NewNoViableAltException(p, nil, nil, nil, nil, nil))
	}

	return localctx
}

// IExprListContext is an interface to support dynamic dispatch.
type IExprListContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Get_expr returns the _expr rule contexts.
	Get_expr() IExprContext

	// Set_expr sets the _expr rule contexts.
	Set_expr(IExprContext)

	// GetE returns the e rule context list.
	GetE() []IExprContext

	// SetE sets the e rule context list.
	SetE([]IExprContext)

	// IsExprListContext differentiates from other interfaces.
	IsExprListContext()
}

type ExprListContext struct {
	*antlr.BaseParserRuleContext
	parser antlr.Parser
	_expr  IExprContext
	e      []IExprContext
}

func NewEmptyExprListContext() *ExprListContext {
	var p = new(ExprListContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_exprList
	return p
}

func (*ExprListContext) IsExprListContext() {}

func NewExprListContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *ExprListContext {
	var p = new(ExprListContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_exprList

	return p
}

func (s *ExprListContext) GetParser() antlr.Parser { return s.parser }

func (s *ExprListContext) Get_expr() IExprContext { return s._expr }

func (s *ExprListContext) Set_expr(v IExprContext) { s._expr = v }

func (s *ExprListContext) GetE() []IExprContext { return s.e }

func (s *ExprListContext) SetE(v []IExprContext) { s.e = v }

func (s *ExprListContext) AllExpr() []IExprContext {
	var ts = s.GetTypedRuleContexts(reflect.TypeOf((*IExprContext)(nil)).Elem())
	var tst = make([]IExprContext, len(ts))

	for i, t := range ts {
		if t != nil {
			tst[i] = t.(IExprContext)
		}
	}

	return tst
}

func (s *ExprListContext) Expr(i int) IExprContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IExprContext)(nil)).Elem(), i)

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *ExprListContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *ExprListContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (s *ExprListContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterExprList(s)
	}
}

func (s *ExprListContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitExprList(s)
	}
}

func (s *ExprListContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitExprList(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) ExprList() (localctx IExprListContext) {
	localctx = NewExprListContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 18, CELParserRULE_exprList)

	defer func() {
		p.ExitRule()
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	var _alt int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(159)

		var _x = p.Expr()

		localctx.(*ExprListContext)._expr = _x
	}
	localctx.(*ExprListContext).e = append(localctx.(*ExprListContext).e, localctx.(*ExprListContext)._expr)
	p.SetState(164)
	p.GetErrorHandler().Sync(p)
	_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 23, p.GetParserRuleContext())

	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			{
				p.SetState(160)
				p.Match(CELParserCOMMA)
			}
			{
				p.SetState(161)

				var _x = p.Expr()

				localctx.(*ExprListContext)._expr = _x
			}
			localctx.(*ExprListContext).e = append(localctx.(*ExprListContext).e, localctx.(*ExprListContext)._expr)

		}
		p.SetState(166)
		p.GetErrorHandler().Sync(p)
		_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 23, p.GetParserRuleContext())
	}

	return localctx
}

// IFieldInitializerListContext is an interface to support dynamic dispatch.
type IFieldInitializerListContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Get_IDENTIFIER returns the _IDENTIFIER token.
	Get_IDENTIFIER() antlr.Token

	// GetS21 returns the s21 token.
	GetS21() antlr.Token

	// Set_IDENTIFIER sets the _IDENTIFIER token.
	Set_IDENTIFIER(antlr.Token)

	// SetS21 sets the s21 token.
	SetS21(antlr.Token)

	// GetFields returns the fields token list.
	GetFields() []antlr.Token

	// GetCols returns the cols token list.
	GetCols() []antlr.Token

	// SetFields sets the fields token list.
	SetFields([]antlr.Token)

	// SetCols sets the cols token list.
	SetCols([]antlr.Token)

	// Get_expr returns the _expr rule contexts.
	Get_expr() IExprContext

	// Set_expr sets the _expr rule contexts.
	Set_expr(IExprContext)

	// GetValues returns the values rule context list.
	GetValues() []IExprContext

	// SetValues sets the values rule context list.
	SetValues([]IExprContext)

	// IsFieldInitializerListContext differentiates from other interfaces.
	IsFieldInitializerListContext()
}

type FieldInitializerListContext struct {
	*antlr.BaseParserRuleContext
	parser      antlr.Parser
	_IDENTIFIER antlr.Token
	fields      []antlr.Token
	s21         antlr.Token
	cols        []antlr.Token
	_expr       IExprContext
	values      []IExprContext
}

func NewEmptyFieldInitializerListContext() *FieldInitializerListContext {
	var p = new(FieldInitializerListContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_fieldInitializerList
	return p
}

func (*FieldInitializerListContext) IsFieldInitializerListContext() {}

func NewFieldInitializerListContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *FieldInitializerListContext {
	var p = new(FieldInitializerListContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_fieldInitializerList

	return p
}

func (s *FieldInitializerListContext) GetParser() antlr.Parser { return s.parser }

func (s *FieldInitializerListContext) Get_IDENTIFIER() antlr.Token { return s._IDENTIFIER }

func (s *FieldInitializerListContext) GetS21() antlr.Token { return s.s21 }

func (s *FieldInitializerListContext) Set_IDENTIFIER(v antlr.Token) { s._IDENTIFIER = v }

func (s *FieldInitializerListContext) SetS21(v antlr.Token) { s.s21 = v }

func (s *FieldInitializerListContext) GetFields() []antlr.Token { return s.fields }

func (s *FieldInitializerListContext) GetCols() []antlr.Token { return s.cols }

func (s *FieldInitializerListContext) SetFields(v []antlr.Token) { s.fields = v }

func (s *FieldInitializerListContext) SetCols(v []antlr.Token) { s.cols = v }

func (s *FieldInitializerListContext) Get_expr() IExprContext { return s._expr }

func (s *FieldInitializerListContext) Set_expr(v IExprContext) { s._expr = v }

func (s *FieldInitializerListContext) GetValues() []IExprContext { return s.values }

func (s *FieldInitializerListContext) SetValues(v []IExprContext) { s.values = v }

func (s *FieldInitializerListContext) AllIDENTIFIER() []antlr.TerminalNode {
	return s.GetTokens(CELParserIDENTIFIER)
}

func (s *FieldInitializerListContext) IDENTIFIER(i int) antlr.TerminalNode {
	return s.GetToken(CELParserIDENTIFIER, i)
}

func (s *FieldInitializerListContext) AllExpr() []IExprContext {
	var ts = s.GetTypedRuleContexts(reflect.TypeOf((*IExprContext)(nil)).Elem())
	var tst = make([]IExprContext, len(ts))

	for i, t := range ts {
		if t != nil {
			tst[i] = t.(IExprContext)
		}
	}

	return tst
}

func (s *FieldInitializerListContext) Expr(i int) IExprContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IExprContext)(nil)).Elem(), i)

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *FieldInitializerListContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *FieldInitializerListContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (s *FieldInitializerListContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterFieldInitializerList(s)
	}
}

func (s *FieldInitializerListContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitFieldInitializerList(s)
	}
}

func (s *FieldInitializerListContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitFieldInitializerList(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) FieldInitializerList() (localctx IFieldInitializerListContext) {
	localctx = NewFieldInitializerListContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 20, CELParserRULE_fieldInitializerList)

	defer func() {
		p.ExitRule()
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	var _alt int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(167)

		var _m = p.Match(CELParserIDENTIFIER)

		localctx.(*FieldInitializerListContext)._IDENTIFIER = _m
	}
	localctx.(*FieldInitializerListContext).fields = append(localctx.(*FieldInitializerListContext).fields, localctx.(*FieldInitializerListContext)._IDENTIFIER)
	{
		p.SetState(168)

		var _m = p.Match(CELParserCOLON)

		localctx.(*FieldInitializerListContext).s21 = _m
	}
	localctx.(*FieldInitializerListContext).cols = append(localctx.(*FieldInitializerListContext).cols, localctx.(*FieldInitializerListContext).s21)
	{
		p.SetState(169)

		var _x = p.Expr()

		localctx.(*FieldInitializerListContext)._expr = _x
	}
	localctx.(*FieldInitializerListContext).values = append(localctx.(*FieldInitializerListContext).values, localctx.(*FieldInitializerListContext)._expr)
	p.SetState(176)
	p.GetErrorHandler().Sync(p)
	_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 24, p.GetParserRuleContext())

	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			{
				p.SetState(170)
				p.Match(CELParserCOMMA)
			}
			{
				p.SetState(171)

				var _m = p.Match(CELParserIDENTIFIER)

				localctx.(*FieldInitializerListContext)._IDENTIFIER = _m
			}
			localctx.(*FieldInitializerListContext).fields = append(localctx.(*FieldInitializerListContext).fields, localctx.(*FieldInitializerListContext)._IDENTIFIER)
			{
				p.SetState(172)

				var _m = p.Match(CELParserCOLON)

				localctx.(*FieldInitializerListContext).s21 = _m
			}
			localctx.(*FieldInitializerListContext).cols = append(localctx.(*FieldInitializerListContext).cols, localctx.(*FieldInitializerListContext).s21)
			{
				p.SetState(173)

				var _x = p.Expr()

				localctx.(*FieldInitializerListContext)._expr = _x
			}
			localctx.(*FieldInitializerListContext).values = append(localctx.(*FieldInitializerListContext).values, localctx.(*FieldInitializerListContext)._expr)

		}
		p.SetState(178)
		p.GetErrorHandler().Sync(p)
		_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 24, p.GetParserRuleContext())
	}

	return localctx
}

// IMapInitializerListContext is an interface to support dynamic dispatch.
type IMapInitializerListContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// GetS21 returns the s21 token.
	GetS21() antlr.Token

	// SetS21 sets the s21 token.
	SetS21(antlr.Token)

	// GetCols returns the cols token list.
	GetCols() []antlr.Token

	// SetCols sets the cols token list.
	SetCols([]antlr.Token)

	// Get_expr returns the _expr rule contexts.
	Get_expr() IExprContext

	// Set_expr sets the _expr rule contexts.
	Set_expr(IExprContext)

	// GetKeys returns the keys rule context list.
	GetKeys() []IExprContext

	// GetValues returns the values rule context list.
	GetValues() []IExprContext

	// SetKeys sets the keys rule context list.
	SetKeys([]IExprContext)

	// SetValues sets the values rule context list.
	SetValues([]IExprContext)

	// IsMapInitializerListContext differentiates from other interfaces.
	IsMapInitializerListContext()
}

type MapInitializerListContext struct {
	*antlr.BaseParserRuleContext
	parser antlr.Parser
	_expr  IExprContext
	keys   []IExprContext
	s21    antlr.Token
	cols   []antlr.Token
	values []IExprContext
}

func NewEmptyMapInitializerListContext() *MapInitializerListContext {
	var p = new(MapInitializerListContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_mapInitializerList
	return p
}

func (*MapInitializerListContext) IsMapInitializerListContext() {}

func NewMapInitializerListContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *MapInitializerListContext {
	var p = new(MapInitializerListContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_mapInitializerList

	return p
}

func (s *MapInitializerListContext) GetParser() antlr.Parser { return s.parser }

func (s *MapInitializerListContext) GetS21() antlr.Token { return s.s21 }

func (s *MapInitializerListContext) SetS21(v antlr.Token) { s.s21 = v }

func (s *MapInitializerListContext) GetCols() []antlr.Token { return s.cols }

func (s *MapInitializerListContext) SetCols(v []antlr.Token) { s.cols = v }

func (s *MapInitializerListContext) Get_expr() IExprContext { return s._expr }

func (s *MapInitializerListContext) Set_expr(v IExprContext) { s._expr = v }

func (s *MapInitializerListContext) GetKeys() []IExprContext { return s.keys }

func (s *MapInitializerListContext) GetValues() []IExprContext { return s.values }

func (s *MapInitializerListContext) SetKeys(v []IExprContext) { s.keys = v }

func (s *MapInitializerListContext) SetValues(v []IExprContext) { s.values = v }

func (s *MapInitializerListContext) AllExpr() []IExprContext {
	var ts = s.GetTypedRuleContexts(reflect.TypeOf((*IExprContext)(nil)).Elem())
	var tst = make([]IExprContext, len(ts))

	for i, t := range ts {
		if t != nil {
			tst[i] = t.(IExprContext)
		}
	}

	return tst
}

func (s *MapInitializerListContext) Expr(i int) IExprContext {
	var t = s.GetTypedRuleContext(reflect.TypeOf((*IExprContext)(nil)).Elem(), i)

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *MapInitializerListContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *MapInitializerListContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (s *MapInitializerListContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterMapInitializerList(s)
	}
}

func (s *MapInitializerListContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitMapInitializerList(s)
	}
}

func (s *MapInitializerListContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitMapInitializerList(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) MapInitializerList() (localctx IMapInitializerListContext) {
	localctx = NewMapInitializerListContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 22, CELParserRULE_mapInitializerList)

	defer func() {
		p.ExitRule()
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	var _alt int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(179)

		var _x = p.Expr()

		localctx.(*MapInitializerListContext)._expr = _x
	}
	localctx.(*MapInitializerListContext).keys = append(localctx.(*MapInitializerListContext).keys, localctx.(*MapInitializerListContext)._expr)
	{
		p.SetState(180)

		var _m = p.Match(CELParserCOLON)

		localctx.(*MapInitializerListContext).s21 = _m
	}
	localctx.(*MapInitializerListContext).cols = append(localctx.(*MapInitializerListContext).cols, localctx.(*MapInitializerListContext).s21)
	{
		p.SetState(181)

		var _x = p.Expr()

		localctx.(*MapInitializerListContext)._expr = _x
	}
	localctx.(*MapInitializerListContext).values = append(localctx.(*MapInitializerListContext).values, localctx.(*MapInitializerListContext)._expr)
	p.SetState(189)
	p.GetErrorHandler().Sync(p)
	_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 25, p.GetParserRuleContext())

	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			{
				p.SetState(182)
				p.Match(CELParserCOMMA)
			}
			{
				p.SetState(183)

				var _x = p.Expr()

				localctx.(*MapInitializerListContext)._expr = _x
			}
			localctx.(*MapInitializerListContext).keys = append(localctx.(*MapInitializerListContext).keys, localctx.(*MapInitializerListContext)._expr)
			{
				p.SetState(184)

				var _m = p.Match(CELParserCOLON)

				localctx.(*MapInitializerListContext).s21 = _m
			}
			localctx.(*MapInitializerListContext).cols = append(localctx.(*MapInitializerListContext).cols, localctx.(*MapInitializerListContext).s21)
			{
				p.SetState(185)

				var _x = p.Expr()

				localctx.(*MapInitializerListContext)._expr = _x
			}
			localctx.(*MapInitializerListContext).values = append(localctx.(*MapInitializerListContext).values, localctx.(*MapInitializerListContext)._expr)

		}
		p.SetState(191)
		p.GetErrorHandler().Sync(p)
		_alt = p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 25, p.GetParserRuleContext())
	}

	return localctx
}

// ILiteralContext is an interface to support dynamic dispatch.
type ILiteralContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// IsLiteralContext differentiates from other interfaces.
	IsLiteralContext()
}

type LiteralContext struct {
	*antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyLiteralContext() *LiteralContext {
	var p = new(LiteralContext)
	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(nil, -1)
	p.RuleIndex = CELParserRULE_literal
	return p
}

func (*LiteralContext) IsLiteralContext() {}

func NewLiteralContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *LiteralContext {
	var p = new(LiteralContext)

	p.BaseParserRuleContext = antlr.NewBaseParserRuleContext(parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_literal

	return p
}

func (s *LiteralContext) GetParser() antlr.Parser { return s.parser }

func (s *LiteralContext) CopyFrom(ctx *LiteralContext) {
	s.BaseParserRuleContext.CopyFrom(ctx.BaseParserRuleContext)
}

func (s *LiteralContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *LiteralContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

type BytesContext struct {
	*LiteralContext
	tok antlr.Token
}

func NewBytesContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *BytesContext {
	var p = new(BytesContext)

	p.LiteralContext = NewEmptyLiteralContext()
	p.parser = parser
	p.CopyFrom(ctx.(*LiteralContext))

	return p
}

func (s *BytesContext) GetTok() antlr.Token { return s.tok }

func (s *BytesContext) SetTok(v antlr.Token) { s.tok = v }

func (s *BytesContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *BytesContext) BYTES() antlr.TerminalNode {
	return s.GetToken(CELParserBYTES, 0)
}

func (s *BytesContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterBytes(s)
	}
}

func (s *BytesContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitBytes(s)
	}
}

func (s *BytesContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitBytes(s)

	default:
		return t.VisitChildren(s)
	}
}

type UintContext struct {
	*LiteralContext
	tok antlr.Token
}

func NewUintContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *UintContext {
	var p = new(UintContext)

	p.LiteralContext = NewEmptyLiteralContext()
	p.parser = parser
	p.CopyFrom(ctx.(*LiteralContext))

	return p
}

func (s *UintContext) GetTok() antlr.Token { return s.tok }

func (s *UintContext) SetTok(v antlr.Token) { s.tok = v }

func (s *UintContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *UintContext) NUM_UINT() antlr.TerminalNode {
	return s.GetToken(CELParserNUM_UINT, 0)
}

func (s *UintContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterUint(s)
	}
}

func (s *UintContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitUint(s)
	}
}

func (s *UintContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitUint(s)

	default:
		return t.VisitChildren(s)
	}
}

type NullContext struct {
	*LiteralContext
	tok antlr.Token
}

func NewNullContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *NullContext {
	var p = new(NullContext)

	p.LiteralContext = NewEmptyLiteralContext()
	p.parser = parser
	p.CopyFrom(ctx.(*LiteralContext))

	return p
}

func (s *NullContext) GetTok() antlr.Token { return s.tok }

func (s *NullContext) SetTok(v antlr.Token) { s.tok = v }

func (s *NullContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *NullContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterNull(s)
	}
}

func (s *NullContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitNull(s)
	}
}

func (s *NullContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitNull(s)

	default:
		return t.VisitChildren(s)
	}
}

type BoolFalseContext struct {
	*LiteralContext
	tok antlr.Token
}

func NewBoolFalseContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *BoolFalseContext {
	var p = new(BoolFalseContext)

	p.LiteralContext = NewEmptyLiteralContext()
	p.parser = parser
	p.CopyFrom(ctx.(*LiteralContext))

	return p
}

func (s *BoolFalseContext) GetTok() antlr.Token { return s.tok }

func (s *BoolFalseContext) SetTok(v antlr.Token) { s.tok = v }

func (s *BoolFalseContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *BoolFalseContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterBoolFalse(s)
	}
}

func (s *BoolFalseContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitBoolFalse(s)
	}
}

func (s *BoolFalseContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitBoolFalse(s)

	default:
		return t.VisitChildren(s)
	}
}

type StringContext struct {
	*LiteralContext
	tok antlr.Token
}

func NewStringContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *StringContext {
	var p = new(StringContext)

	p.LiteralContext = NewEmptyLiteralContext()
	p.parser = parser
	p.CopyFrom(ctx.(*LiteralContext))

	return p
}

func (s *StringContext) GetTok() antlr.Token { return s.tok }

func (s *StringContext) SetTok(v antlr.Token) { s.tok = v }

func (s *StringContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *StringContext) STRING() antlr.TerminalNode {
	return s.GetToken(CELParserSTRING, 0)
}

func (s *StringContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterString(s)
	}
}

func (s *StringContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitString(s)
	}
}

func (s *StringContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitString(s)

	default:
		return t.VisitChildren(s)
	}
}

type DoubleContext struct {
	*LiteralContext
	sign antlr.Token
	tok  antlr.Token
}

func NewDoubleContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *DoubleContext {
	var p = new(DoubleContext)

	p.LiteralContext = NewEmptyLiteralContext()
	p.parser = parser
	p.CopyFrom(ctx.(*LiteralContext))

	return p
}

func (s *DoubleContext) GetSign() antlr.Token { return s.sign }

func (s *DoubleContext) GetTok() antlr.Token { return s.tok }

func (s *DoubleContext) SetSign(v antlr.Token) { s.sign = v }

func (s *DoubleContext) SetTok(v antlr.Token) { s.tok = v }

func (s *DoubleContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *DoubleContext) NUM_FLOAT() antlr.TerminalNode {
	return s.GetToken(CELParserNUM_FLOAT, 0)
}

func (s *DoubleContext) MINUS() antlr.TerminalNode {
	return s.GetToken(CELParserMINUS, 0)
}

func (s *DoubleContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterDouble(s)
	}
}

func (s *DoubleContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitDouble(s)
	}
}

func (s *DoubleContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitDouble(s)

	default:
		return t.VisitChildren(s)
	}
}

type BoolTrueContext struct {
	*LiteralContext
	tok antlr.Token
}

func NewBoolTrueContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *BoolTrueContext {
	var p = new(BoolTrueContext)

	p.LiteralContext = NewEmptyLiteralContext()
	p.parser = parser
	p.CopyFrom(ctx.(*LiteralContext))

	return p
}

func (s *BoolTrueContext) GetTok() antlr.Token { return s.tok }

func (s *BoolTrueContext) SetTok(v antlr.Token) { s.tok = v }

func (s *BoolTrueContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *BoolTrueContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterBoolTrue(s)
	}
}

func (s *BoolTrueContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitBoolTrue(s)
	}
}

func (s *BoolTrueContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitBoolTrue(s)

	default:
		return t.VisitChildren(s)
	}
}

type IntContext struct {
	*LiteralContext
	sign antlr.Token
	tok  antlr.Token
}

func NewIntContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *IntContext {
	var p = new(IntContext)

	p.LiteralContext = NewEmptyLiteralContext()
	p.parser = parser
	p.CopyFrom(ctx.(*LiteralContext))

	return p
}

func (s *IntContext) GetSign() antlr.Token { return s.sign }

func (s *IntContext) GetTok() antlr.Token { return s.tok }

func (s *IntContext) SetSign(v antlr.Token) { s.sign = v }

func (s *IntContext) SetTok(v antlr.Token) { s.tok = v }

func (s *IntContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *IntContext) NUM_INT() antlr.TerminalNode {
	return s.GetToken(CELParserNUM_INT, 0)
}

func (s *IntContext) MINUS() antlr.TerminalNode {
	return s.GetToken(CELParserMINUS, 0)
}

func (s *IntContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterInt(s)
	}
}

func (s *IntContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitInt(s)
	}
}

func (s *IntContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitInt(s)

	default:
		return t.VisitChildren(s)
	}
}

func (p *CELParser) Literal() (localctx ILiteralContext) {
	localctx = NewLiteralContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 24, CELParserRULE_literal)
	var _la int

	defer func() {
		p.ExitRule()
	}()

	defer func() {
		if err := recover(); err != nil {
			if v, ok := err.(antlr.RecognitionException); ok {
				localctx.SetException(v)
				p.GetErrorHandler().ReportError(p, v)
				p.GetErrorHandler().Recover(p, v)
			} else {
				panic(err)
			}
		}
	}()

	p.SetState(206)
	p.GetErrorHandler().Sync(p)
	switch p.GetInterpreter().AdaptivePredict(p.GetTokenStream(), 28, p.GetParserRuleContext()) {
	case 1:
		localctx = NewIntContext(p, localctx)
		p.EnterOuterAlt(localctx, 1)
		p.SetState(193)
		p.GetErrorHandler().Sync(p)
		_la = p.GetTokenStream().LA(1)

		if _la == CELParserMINUS {
			{
				p.SetState(192)

				var _m = p.Match(CELParserMINUS)

				localctx.(*IntContext).sign = _m
			}

		}
		{
			p.SetState(195)

			var _m = p.Match(CELParserNUM_INT)

			localctx.(*IntContext).tok = _m
		}

	case 2:
		localctx = NewUintContext(p, localctx)
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(196)

			var _m = p.Match(CELParserNUM_UINT)

			localctx.(*UintContext).tok = _m
		}

	case 3:
		localctx = NewDoubleContext(p, localctx)
		p.EnterOuterAlt(localctx, 3)
		p.SetState(198)
		p.GetErrorHandler().Sync(p)
		_la = p.GetTokenStream().LA(1)

		if _la == CELParserMINUS {
			{
				p.SetState(197)

				var _m = p.Match(CELParserMINUS)

				localctx.(*DoubleContext).sign = _m
			}

		}
		{
			p.SetState(200)

			var _m = p.Match(CELParserNUM_FLOAT)

			localctx.(*DoubleContext).tok = _m
		}

	case 4:
		localctx = NewStringContext(p, localctx)
		p.EnterOuterAlt(localctx, 4)
		{
			p.SetState(201)

			var _m = p.Match(CELParserSTRING)

			localctx.(*StringContext).tok = _m
		}

	case 5:
		localctx = NewBytesContext(p, localctx)
		p.EnterOuterAlt(localctx, 5)
		{
			p.SetState(202)

			var _m = p.Match(CELParserBYTES)

			localctx.(*BytesContext).tok = _m
		}

	case 6:
		localctx = NewBoolTrueContext(p, localctx)
		p.EnterOuterAlt(localctx, 6)
		{
			p.SetState(203)

			var _m = p.Match(CELParserTRUE)

			localctx.(*BoolTrueContext).tok = _m
		}

	case 7:
		localctx = NewBoolFalseContext(p, localctx)
		p.EnterOuterAlt(localctx, 7)
		{
			p.SetState(204)

			var _m = p.Match(CELParserFALSE)

			localctx.(*BoolFalseContext).tok = _m
		}

	case 8:
		localctx = NewNullContext(p, localctx)
		p.EnterOuterAlt(localctx, 8)
		{
			p.SetState(205)

			var _m = p.Match(CELParserNULL)

			localctx.(*NullContext).tok = _m
		}

	}

	return localctx
}

func (p *CELParser) Sempred(localctx antlr.RuleContext, ruleIndex, predIndex int) bool {
	switch ruleIndex {
	case 4:
		var t *RelationContext = nil
		if localctx != nil {
			t = localctx.(*RelationContext)
		}
		return p.Relation_Sempred(t, predIndex)

	case 5:
		var t *CalcContext = nil
		if localctx != nil {
			t = localctx.(*CalcContext)
		}
		return p.Calc_Sempred(t, predIndex)

	case 7:
		var t *MemberContext = nil
		if localctx != nil {
			t = localctx.(*MemberContext)
		}
		return p.Member_Sempred(t, predIndex)

	default:
		panic("No predicate with index: " + fmt.Sprint(ruleIndex))
	}
}

func (p *CELParser) Relation_Sempred(localctx antlr.RuleContext, predIndex int) bool {
	switch predIndex {
	case 0:
		return p.Precpred(p.GetParserRuleContext(), 1)

	default:
		panic("No predicate with index: " + fmt.Sprint(predIndex))
	}
}

func (p *CELParser) Calc_Sempred(localctx antlr.RuleContext, predIndex int) bool {
	switch predIndex {
	case 1:
		return p.Precpred(p.GetParserRuleContext(), 2)

	case 2:
		return p.Precpred(p.GetParserRuleContext(), 1)

	default:
		panic("No predicate with index: " + fmt.Sprint(predIndex))
	}
}

func (p *CELParser) Member_Sempred(localctx antlr.RuleContext, predIndex int) bool {
	switch predIndex {
	case 3:
		return p.Precpred(p.GetParserRuleContext(), 3)

	case 4:
		return p.Precpred(p.GetParserRuleContext(), 2)

	case 5:
		return p.Precpred(p.GetParserRuleContext(), 1)

	default:
		panic("No predicate with index: " + fmt.Sprint(predIndex))
	}
}
