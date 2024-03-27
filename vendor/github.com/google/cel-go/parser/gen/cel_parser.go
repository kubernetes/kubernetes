// Code generated from /usr/local/google/home/tswadell/go/src/github.com/google/cel-go/parser/gen/CEL.g4 by ANTLR 4.13.1. DO NOT EDIT.

package gen // CEL
import (
	"fmt"
	"strconv"
  	"sync"

	"github.com/antlr4-go/antlr/v4"
)

// Suppress unused import errors
var _ = fmt.Printf
var _ = strconv.Itoa
var _ = sync.Once{}


type CELParser struct {
	*antlr.BaseParser
}

var CELParserStaticData struct {
  once                   sync.Once
  serializedATN          []int32
  LiteralNames           []string
  SymbolicNames          []string
  RuleNames              []string
  PredictionContextCache *antlr.PredictionContextCache
  atn                    *antlr.ATN
  decisionToDFA          []*antlr.DFA
}

func celParserInit() {
  staticData := &CELParserStaticData
  staticData.LiteralNames = []string{
    "", "'=='", "'!='", "'in'", "'<'", "'<='", "'>='", "'>'", "'&&'", "'||'", 
    "'['", "']'", "'{'", "'}'", "'('", "')'", "'.'", "','", "'-'", "'!'", 
    "'?'", "':'", "'+'", "'*'", "'/'", "'%'", "'true'", "'false'", "'null'",
  }
  staticData.SymbolicNames = []string{
    "", "EQUALS", "NOT_EQUALS", "IN", "LESS", "LESS_EQUALS", "GREATER_EQUALS", 
    "GREATER", "LOGICAL_AND", "LOGICAL_OR", "LBRACKET", "RPRACKET", "LBRACE", 
    "RBRACE", "LPAREN", "RPAREN", "DOT", "COMMA", "MINUS", "EXCLAM", "QUESTIONMARK", 
    "COLON", "PLUS", "STAR", "SLASH", "PERCENT", "CEL_TRUE", "CEL_FALSE", 
    "NUL", "WHITESPACE", "COMMENT", "NUM_FLOAT", "NUM_INT", "NUM_UINT", 
    "STRING", "BYTES", "IDENTIFIER",
  }
  staticData.RuleNames = []string{
    "start", "expr", "conditionalOr", "conditionalAnd", "relation", "calc", 
    "unary", "member", "primary", "exprList", "listInit", "fieldInitializerList", 
    "optField", "mapInitializerList", "optExpr", "literal",
  }
  staticData.PredictionContextCache = antlr.NewPredictionContextCache()
  staticData.serializedATN = []int32{
	4, 1, 36, 251, 2, 0, 7, 0, 2, 1, 7, 1, 2, 2, 7, 2, 2, 3, 7, 3, 2, 4, 7, 
	4, 2, 5, 7, 5, 2, 6, 7, 6, 2, 7, 7, 7, 2, 8, 7, 8, 2, 9, 7, 9, 2, 10, 7, 
	10, 2, 11, 7, 11, 2, 12, 7, 12, 2, 13, 7, 13, 2, 14, 7, 14, 2, 15, 7, 15, 
	1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 42, 8, 1, 1, 
	2, 1, 2, 1, 2, 5, 2, 47, 8, 2, 10, 2, 12, 2, 50, 9, 2, 1, 3, 1, 3, 1, 3, 
	5, 3, 55, 8, 3, 10, 3, 12, 3, 58, 9, 3, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 
	4, 5, 4, 66, 8, 4, 10, 4, 12, 4, 69, 9, 4, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 
	1, 5, 1, 5, 1, 5, 1, 5, 5, 5, 80, 8, 5, 10, 5, 12, 5, 83, 9, 5, 1, 6, 1, 
	6, 4, 6, 87, 8, 6, 11, 6, 12, 6, 88, 1, 6, 1, 6, 4, 6, 93, 8, 6, 11, 6, 
	12, 6, 94, 1, 6, 3, 6, 98, 8, 6, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 3, 
	7, 106, 8, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 3, 7, 114, 8, 7, 1, 7, 
	1, 7, 1, 7, 1, 7, 3, 7, 120, 8, 7, 1, 7, 1, 7, 1, 7, 5, 7, 125, 8, 7, 10, 
	7, 12, 7, 128, 9, 7, 1, 8, 3, 8, 131, 8, 8, 1, 8, 1, 8, 1, 8, 3, 8, 136, 
	8, 8, 1, 8, 3, 8, 139, 8, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 3, 8, 
	147, 8, 8, 1, 8, 3, 8, 150, 8, 8, 1, 8, 1, 8, 1, 8, 3, 8, 155, 8, 8, 1, 
	8, 3, 8, 158, 8, 8, 1, 8, 1, 8, 3, 8, 162, 8, 8, 1, 8, 1, 8, 1, 8, 5, 8, 
	167, 8, 8, 10, 8, 12, 8, 170, 9, 8, 1, 8, 1, 8, 3, 8, 174, 8, 8, 1, 8, 
	3, 8, 177, 8, 8, 1, 8, 1, 8, 3, 8, 181, 8, 8, 1, 9, 1, 9, 1, 9, 5, 9, 186, 
	8, 9, 10, 9, 12, 9, 189, 9, 9, 1, 10, 1, 10, 1, 10, 5, 10, 194, 8, 10, 
	10, 10, 12, 10, 197, 9, 10, 1, 11, 1, 11, 1, 11, 1, 11, 1, 11, 1, 11, 1, 
	11, 1, 11, 5, 11, 207, 8, 11, 10, 11, 12, 11, 210, 9, 11, 1, 12, 3, 12, 
	213, 8, 12, 1, 12, 1, 12, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 
	13, 1, 13, 5, 13, 225, 8, 13, 10, 13, 12, 13, 228, 9, 13, 1, 14, 3, 14, 
	231, 8, 14, 1, 14, 1, 14, 1, 15, 3, 15, 236, 8, 15, 1, 15, 1, 15, 1, 15, 
	3, 15, 241, 8, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 3, 15, 249, 
	8, 15, 1, 15, 0, 3, 8, 10, 14, 16, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 
	22, 24, 26, 28, 30, 0, 3, 1, 0, 1, 7, 1, 0, 23, 25, 2, 0, 18, 18, 22, 22, 
	281, 0, 32, 1, 0, 0, 0, 2, 35, 1, 0, 0, 0, 4, 43, 1, 0, 0, 0, 6, 51, 1, 
	0, 0, 0, 8, 59, 1, 0, 0, 0, 10, 70, 1, 0, 0, 0, 12, 97, 1, 0, 0, 0, 14, 
	99, 1, 0, 0, 0, 16, 180, 1, 0, 0, 0, 18, 182, 1, 0, 0, 0, 20, 190, 1, 0, 
	0, 0, 22, 198, 1, 0, 0, 0, 24, 212, 1, 0, 0, 0, 26, 216, 1, 0, 0, 0, 28, 
	230, 1, 0, 0, 0, 30, 248, 1, 0, 0, 0, 32, 33, 3, 2, 1, 0, 33, 34, 5, 0, 
	0, 1, 34, 1, 1, 0, 0, 0, 35, 41, 3, 4, 2, 0, 36, 37, 5, 20, 0, 0, 37, 38, 
	3, 4, 2, 0, 38, 39, 5, 21, 0, 0, 39, 40, 3, 2, 1, 0, 40, 42, 1, 0, 0, 0, 
	41, 36, 1, 0, 0, 0, 41, 42, 1, 0, 0, 0, 42, 3, 1, 0, 0, 0, 43, 48, 3, 6, 
	3, 0, 44, 45, 5, 9, 0, 0, 45, 47, 3, 6, 3, 0, 46, 44, 1, 0, 0, 0, 47, 50, 
	1, 0, 0, 0, 48, 46, 1, 0, 0, 0, 48, 49, 1, 0, 0, 0, 49, 5, 1, 0, 0, 0, 
	50, 48, 1, 0, 0, 0, 51, 56, 3, 8, 4, 0, 52, 53, 5, 8, 0, 0, 53, 55, 3, 
	8, 4, 0, 54, 52, 1, 0, 0, 0, 55, 58, 1, 0, 0, 0, 56, 54, 1, 0, 0, 0, 56, 
	57, 1, 0, 0, 0, 57, 7, 1, 0, 0, 0, 58, 56, 1, 0, 0, 0, 59, 60, 6, 4, -1, 
	0, 60, 61, 3, 10, 5, 0, 61, 67, 1, 0, 0, 0, 62, 63, 10, 1, 0, 0, 63, 64, 
	7, 0, 0, 0, 64, 66, 3, 8, 4, 2, 65, 62, 1, 0, 0, 0, 66, 69, 1, 0, 0, 0, 
	67, 65, 1, 0, 0, 0, 67, 68, 1, 0, 0, 0, 68, 9, 1, 0, 0, 0, 69, 67, 1, 0, 
	0, 0, 70, 71, 6, 5, -1, 0, 71, 72, 3, 12, 6, 0, 72, 81, 1, 0, 0, 0, 73, 
	74, 10, 2, 0, 0, 74, 75, 7, 1, 0, 0, 75, 80, 3, 10, 5, 3, 76, 77, 10, 1, 
	0, 0, 77, 78, 7, 2, 0, 0, 78, 80, 3, 10, 5, 2, 79, 73, 1, 0, 0, 0, 79, 
	76, 1, 0, 0, 0, 80, 83, 1, 0, 0, 0, 81, 79, 1, 0, 0, 0, 81, 82, 1, 0, 0, 
	0, 82, 11, 1, 0, 0, 0, 83, 81, 1, 0, 0, 0, 84, 98, 3, 14, 7, 0, 85, 87, 
	5, 19, 0, 0, 86, 85, 1, 0, 0, 0, 87, 88, 1, 0, 0, 0, 88, 86, 1, 0, 0, 0, 
	88, 89, 1, 0, 0, 0, 89, 90, 1, 0, 0, 0, 90, 98, 3, 14, 7, 0, 91, 93, 5, 
	18, 0, 0, 92, 91, 1, 0, 0, 0, 93, 94, 1, 0, 0, 0, 94, 92, 1, 0, 0, 0, 94, 
	95, 1, 0, 0, 0, 95, 96, 1, 0, 0, 0, 96, 98, 3, 14, 7, 0, 97, 84, 1, 0, 
	0, 0, 97, 86, 1, 0, 0, 0, 97, 92, 1, 0, 0, 0, 98, 13, 1, 0, 0, 0, 99, 100, 
	6, 7, -1, 0, 100, 101, 3, 16, 8, 0, 101, 126, 1, 0, 0, 0, 102, 103, 10, 
	3, 0, 0, 103, 105, 5, 16, 0, 0, 104, 106, 5, 20, 0, 0, 105, 104, 1, 0, 
	0, 0, 105, 106, 1, 0, 0, 0, 106, 107, 1, 0, 0, 0, 107, 125, 5, 36, 0, 0, 
	108, 109, 10, 2, 0, 0, 109, 110, 5, 16, 0, 0, 110, 111, 5, 36, 0, 0, 111, 
	113, 5, 14, 0, 0, 112, 114, 3, 18, 9, 0, 113, 112, 1, 0, 0, 0, 113, 114, 
	1, 0, 0, 0, 114, 115, 1, 0, 0, 0, 115, 125, 5, 15, 0, 0, 116, 117, 10, 
	1, 0, 0, 117, 119, 5, 10, 0, 0, 118, 120, 5, 20, 0, 0, 119, 118, 1, 0, 
	0, 0, 119, 120, 1, 0, 0, 0, 120, 121, 1, 0, 0, 0, 121, 122, 3, 2, 1, 0, 
	122, 123, 5, 11, 0, 0, 123, 125, 1, 0, 0, 0, 124, 102, 1, 0, 0, 0, 124, 
	108, 1, 0, 0, 0, 124, 116, 1, 0, 0, 0, 125, 128, 1, 0, 0, 0, 126, 124, 
	1, 0, 0, 0, 126, 127, 1, 0, 0, 0, 127, 15, 1, 0, 0, 0, 128, 126, 1, 0, 
	0, 0, 129, 131, 5, 16, 0, 0, 130, 129, 1, 0, 0, 0, 130, 131, 1, 0, 0, 0, 
	131, 132, 1, 0, 0, 0, 132, 138, 5, 36, 0, 0, 133, 135, 5, 14, 0, 0, 134, 
	136, 3, 18, 9, 0, 135, 134, 1, 0, 0, 0, 135, 136, 1, 0, 0, 0, 136, 137, 
	1, 0, 0, 0, 137, 139, 5, 15, 0, 0, 138, 133, 1, 0, 0, 0, 138, 139, 1, 0, 
	0, 0, 139, 181, 1, 0, 0, 0, 140, 141, 5, 14, 0, 0, 141, 142, 3, 2, 1, 0, 
	142, 143, 5, 15, 0, 0, 143, 181, 1, 0, 0, 0, 144, 146, 5, 10, 0, 0, 145, 
	147, 3, 20, 10, 0, 146, 145, 1, 0, 0, 0, 146, 147, 1, 0, 0, 0, 147, 149, 
	1, 0, 0, 0, 148, 150, 5, 17, 0, 0, 149, 148, 1, 0, 0, 0, 149, 150, 1, 0, 
	0, 0, 150, 151, 1, 0, 0, 0, 151, 181, 5, 11, 0, 0, 152, 154, 5, 12, 0, 
	0, 153, 155, 3, 26, 13, 0, 154, 153, 1, 0, 0, 0, 154, 155, 1, 0, 0, 0, 
	155, 157, 1, 0, 0, 0, 156, 158, 5, 17, 0, 0, 157, 156, 1, 0, 0, 0, 157, 
	158, 1, 0, 0, 0, 158, 159, 1, 0, 0, 0, 159, 181, 5, 13, 0, 0, 160, 162, 
	5, 16, 0, 0, 161, 160, 1, 0, 0, 0, 161, 162, 1, 0, 0, 0, 162, 163, 1, 0, 
	0, 0, 163, 168, 5, 36, 0, 0, 164, 165, 5, 16, 0, 0, 165, 167, 5, 36, 0, 
	0, 166, 164, 1, 0, 0, 0, 167, 170, 1, 0, 0, 0, 168, 166, 1, 0, 0, 0, 168, 
	169, 1, 0, 0, 0, 169, 171, 1, 0, 0, 0, 170, 168, 1, 0, 0, 0, 171, 173, 
	5, 12, 0, 0, 172, 174, 3, 22, 11, 0, 173, 172, 1, 0, 0, 0, 173, 174, 1, 
	0, 0, 0, 174, 176, 1, 0, 0, 0, 175, 177, 5, 17, 0, 0, 176, 175, 1, 0, 0, 
	0, 176, 177, 1, 0, 0, 0, 177, 178, 1, 0, 0, 0, 178, 181, 5, 13, 0, 0, 179, 
	181, 3, 30, 15, 0, 180, 130, 1, 0, 0, 0, 180, 140, 1, 0, 0, 0, 180, 144, 
	1, 0, 0, 0, 180, 152, 1, 0, 0, 0, 180, 161, 1, 0, 0, 0, 180, 179, 1, 0, 
	0, 0, 181, 17, 1, 0, 0, 0, 182, 187, 3, 2, 1, 0, 183, 184, 5, 17, 0, 0, 
	184, 186, 3, 2, 1, 0, 185, 183, 1, 0, 0, 0, 186, 189, 1, 0, 0, 0, 187, 
	185, 1, 0, 0, 0, 187, 188, 1, 0, 0, 0, 188, 19, 1, 0, 0, 0, 189, 187, 1, 
	0, 0, 0, 190, 195, 3, 28, 14, 0, 191, 192, 5, 17, 0, 0, 192, 194, 3, 28, 
	14, 0, 193, 191, 1, 0, 0, 0, 194, 197, 1, 0, 0, 0, 195, 193, 1, 0, 0, 0, 
	195, 196, 1, 0, 0, 0, 196, 21, 1, 0, 0, 0, 197, 195, 1, 0, 0, 0, 198, 199, 
	3, 24, 12, 0, 199, 200, 5, 21, 0, 0, 200, 208, 3, 2, 1, 0, 201, 202, 5, 
	17, 0, 0, 202, 203, 3, 24, 12, 0, 203, 204, 5, 21, 0, 0, 204, 205, 3, 2, 
	1, 0, 205, 207, 1, 0, 0, 0, 206, 201, 1, 0, 0, 0, 207, 210, 1, 0, 0, 0, 
	208, 206, 1, 0, 0, 0, 208, 209, 1, 0, 0, 0, 209, 23, 1, 0, 0, 0, 210, 208, 
	1, 0, 0, 0, 211, 213, 5, 20, 0, 0, 212, 211, 1, 0, 0, 0, 212, 213, 1, 0, 
	0, 0, 213, 214, 1, 0, 0, 0, 214, 215, 5, 36, 0, 0, 215, 25, 1, 0, 0, 0, 
	216, 217, 3, 28, 14, 0, 217, 218, 5, 21, 0, 0, 218, 226, 3, 2, 1, 0, 219, 
	220, 5, 17, 0, 0, 220, 221, 3, 28, 14, 0, 221, 222, 5, 21, 0, 0, 222, 223, 
	3, 2, 1, 0, 223, 225, 1, 0, 0, 0, 224, 219, 1, 0, 0, 0, 225, 228, 1, 0, 
	0, 0, 226, 224, 1, 0, 0, 0, 226, 227, 1, 0, 0, 0, 227, 27, 1, 0, 0, 0, 
	228, 226, 1, 0, 0, 0, 229, 231, 5, 20, 0, 0, 230, 229, 1, 0, 0, 0, 230, 
	231, 1, 0, 0, 0, 231, 232, 1, 0, 0, 0, 232, 233, 3, 2, 1, 0, 233, 29, 1, 
	0, 0, 0, 234, 236, 5, 18, 0, 0, 235, 234, 1, 0, 0, 0, 235, 236, 1, 0, 0, 
	0, 236, 237, 1, 0, 0, 0, 237, 249, 5, 32, 0, 0, 238, 249, 5, 33, 0, 0, 
	239, 241, 5, 18, 0, 0, 240, 239, 1, 0, 0, 0, 240, 241, 1, 0, 0, 0, 241, 
	242, 1, 0, 0, 0, 242, 249, 5, 31, 0, 0, 243, 249, 5, 34, 0, 0, 244, 249, 
	5, 35, 0, 0, 245, 249, 5, 26, 0, 0, 246, 249, 5, 27, 0, 0, 247, 249, 5, 
	28, 0, 0, 248, 235, 1, 0, 0, 0, 248, 238, 1, 0, 0, 0, 248, 240, 1, 0, 0, 
	0, 248, 243, 1, 0, 0, 0, 248, 244, 1, 0, 0, 0, 248, 245, 1, 0, 0, 0, 248, 
	246, 1, 0, 0, 0, 248, 247, 1, 0, 0, 0, 249, 31, 1, 0, 0, 0, 35, 41, 48, 
	56, 67, 79, 81, 88, 94, 97, 105, 113, 119, 124, 126, 130, 135, 138, 146, 
	149, 154, 157, 161, 168, 173, 176, 180, 187, 195, 208, 212, 226, 230, 235, 
	240, 248,
}
  deserializer := antlr.NewATNDeserializer(nil)
  staticData.atn = deserializer.Deserialize(staticData.serializedATN)
  atn := staticData.atn
  staticData.decisionToDFA = make([]*antlr.DFA, len(atn.DecisionToState))
  decisionToDFA := staticData.decisionToDFA
  for index, state := range atn.DecisionToState {
    decisionToDFA[index] = antlr.NewDFA(state, index)
  }
}

// CELParserInit initializes any static state used to implement CELParser. By default the
// static state used to implement the parser is lazily initialized during the first call to
// NewCELParser(). You can call this function if you wish to initialize the static state ahead
// of time.
func CELParserInit() {
  staticData := &CELParserStaticData
  staticData.once.Do(celParserInit)
}

// NewCELParser produces a new parser instance for the optional input antlr.TokenStream.
func NewCELParser(input antlr.TokenStream) *CELParser {
	CELParserInit()
	this := new(CELParser)
	this.BaseParser = antlr.NewBaseParser(input)
  staticData := &CELParserStaticData
	this.Interpreter = antlr.NewParserATNSimulator(this, staticData.atn, staticData.decisionToDFA, staticData.PredictionContextCache)
	this.RuleNames = staticData.RuleNames
	this.LiteralNames = staticData.LiteralNames
	this.SymbolicNames = staticData.SymbolicNames
	this.GrammarFileName = "CEL.g4"

	return this
}


// CELParser tokens.
const (
	CELParserEOF = antlr.TokenEOF
	CELParserEQUALS = 1
	CELParserNOT_EQUALS = 2
	CELParserIN = 3
	CELParserLESS = 4
	CELParserLESS_EQUALS = 5
	CELParserGREATER_EQUALS = 6
	CELParserGREATER = 7
	CELParserLOGICAL_AND = 8
	CELParserLOGICAL_OR = 9
	CELParserLBRACKET = 10
	CELParserRPRACKET = 11
	CELParserLBRACE = 12
	CELParserRBRACE = 13
	CELParserLPAREN = 14
	CELParserRPAREN = 15
	CELParserDOT = 16
	CELParserCOMMA = 17
	CELParserMINUS = 18
	CELParserEXCLAM = 19
	CELParserQUESTIONMARK = 20
	CELParserCOLON = 21
	CELParserPLUS = 22
	CELParserSTAR = 23
	CELParserSLASH = 24
	CELParserPERCENT = 25
	CELParserCEL_TRUE = 26
	CELParserCEL_FALSE = 27
	CELParserNUL = 28
	CELParserWHITESPACE = 29
	CELParserCOMMENT = 30
	CELParserNUM_FLOAT = 31
	CELParserNUM_INT = 32
	CELParserNUM_UINT = 33
	CELParserSTRING = 34
	CELParserBYTES = 35
	CELParserIDENTIFIER = 36
)

// CELParser rules.
const (
	CELParserRULE_start = 0
	CELParserRULE_expr = 1
	CELParserRULE_conditionalOr = 2
	CELParserRULE_conditionalAnd = 3
	CELParserRULE_relation = 4
	CELParserRULE_calc = 5
	CELParserRULE_unary = 6
	CELParserRULE_member = 7
	CELParserRULE_primary = 8
	CELParserRULE_exprList = 9
	CELParserRULE_listInit = 10
	CELParserRULE_fieldInitializerList = 11
	CELParserRULE_optField = 12
	CELParserRULE_mapInitializerList = 13
	CELParserRULE_optExpr = 14
	CELParserRULE_literal = 15
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


	// Getter signatures
	EOF() antlr.TerminalNode
	Expr() IExprContext

	// IsStartContext differentiates from other interfaces.
	IsStartContext()
}

type StartContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	e IExprContext 
}

func NewEmptyStartContext() *StartContext {
	var p = new(StartContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_start
	return p
}

func InitEmptyStartContext(p *StartContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_start
}

func (*StartContext) IsStartContext() {}

func NewStartContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *StartContext {
	var p = new(StartContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

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
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IExprContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

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




func (p *CELParser) Start_() (localctx IStartContext) {
	localctx = NewStartContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 0, CELParserRULE_start)
	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(32)

		var _x = p.Expr()


		localctx.(*StartContext).e = _x
	}
	{
		p.SetState(33)
		p.Match(CELParserEOF)
		if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
		}
	}



errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
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


	// Getter signatures
	AllConditionalOr() []IConditionalOrContext
	ConditionalOr(i int) IConditionalOrContext
	COLON() antlr.TerminalNode
	QUESTIONMARK() antlr.TerminalNode
	Expr() IExprContext

	// IsExprContext differentiates from other interfaces.
	IsExprContext()
}

type ExprContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	e IConditionalOrContext 
	op antlr.Token
	e1 IConditionalOrContext 
	e2 IExprContext 
}

func NewEmptyExprContext() *ExprContext {
	var p = new(ExprContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_expr
	return p
}

func InitEmptyExprContext(p *ExprContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_expr
}

func (*ExprContext) IsExprContext() {}

func NewExprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *ExprContext {
	var p = new(ExprContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

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
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IConditionalOrContext); ok {
			len++
		}
	}

	tst := make([]IConditionalOrContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IConditionalOrContext); ok {
			tst[i] = t.(IConditionalOrContext)
			i++
		}
	}

	return tst
}

func (s *ExprContext) ConditionalOr(i int) IConditionalOrContext {
	var t antlr.RuleContext;
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IConditionalOrContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext);
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IConditionalOrContext)
}

func (s *ExprContext) COLON() antlr.TerminalNode {
	return s.GetToken(CELParserCOLON, 0)
}

func (s *ExprContext) QUESTIONMARK() antlr.TerminalNode {
	return s.GetToken(CELParserQUESTIONMARK, 0)
}

func (s *ExprContext) Expr() IExprContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IExprContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

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

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(35)

		var _x = p.ConditionalOr()


		localctx.(*ExprContext).e = _x
	}
	p.SetState(41)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_la = p.GetTokenStream().LA(1)


	if _la == CELParserQUESTIONMARK {
		{
			p.SetState(36)

			var _m = p.Match(CELParserQUESTIONMARK)

			localctx.(*ExprContext).op = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}
		{
			p.SetState(37)

			var _x = p.ConditionalOr()


			localctx.(*ExprContext).e1 = _x
		}
		{
			p.SetState(38)
			p.Match(CELParserCOLON)
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}
		{
			p.SetState(39)

			var _x = p.Expr()


			localctx.(*ExprContext).e2 = _x
		}

	}



errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
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


	// Getter signatures
	AllConditionalAnd() []IConditionalAndContext
	ConditionalAnd(i int) IConditionalAndContext
	AllLOGICAL_OR() []antlr.TerminalNode
	LOGICAL_OR(i int) antlr.TerminalNode

	// IsConditionalOrContext differentiates from other interfaces.
	IsConditionalOrContext()
}

type ConditionalOrContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	e IConditionalAndContext 
	s9 antlr.Token
	ops []antlr.Token
	_conditionalAnd IConditionalAndContext 
	e1 []IConditionalAndContext
}

func NewEmptyConditionalOrContext() *ConditionalOrContext {
	var p = new(ConditionalOrContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_conditionalOr
	return p
}

func InitEmptyConditionalOrContext(p *ConditionalOrContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_conditionalOr
}

func (*ConditionalOrContext) IsConditionalOrContext() {}

func NewConditionalOrContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *ConditionalOrContext {
	var p = new(ConditionalOrContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

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
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IConditionalAndContext); ok {
			len++
		}
	}

	tst := make([]IConditionalAndContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IConditionalAndContext); ok {
			tst[i] = t.(IConditionalAndContext)
			i++
		}
	}

	return tst
}

func (s *ConditionalOrContext) ConditionalAnd(i int) IConditionalAndContext {
	var t antlr.RuleContext;
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IConditionalAndContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext);
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IConditionalAndContext)
}

func (s *ConditionalOrContext) AllLOGICAL_OR() []antlr.TerminalNode {
	return s.GetTokens(CELParserLOGICAL_OR)
}

func (s *ConditionalOrContext) LOGICAL_OR(i int) antlr.TerminalNode {
	return s.GetToken(CELParserLOGICAL_OR, i)
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

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(43)

		var _x = p.ConditionalAnd()


		localctx.(*ConditionalOrContext).e = _x
	}
	p.SetState(48)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_la = p.GetTokenStream().LA(1)


	for _la == CELParserLOGICAL_OR {
		{
			p.SetState(44)

			var _m = p.Match(CELParserLOGICAL_OR)

			localctx.(*ConditionalOrContext).s9 = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}
		localctx.(*ConditionalOrContext).ops = append(localctx.(*ConditionalOrContext).ops, localctx.(*ConditionalOrContext).s9)
		{
			p.SetState(45)

			var _x = p.ConditionalAnd()


			localctx.(*ConditionalOrContext)._conditionalAnd = _x
		}
		localctx.(*ConditionalOrContext).e1 = append(localctx.(*ConditionalOrContext).e1, localctx.(*ConditionalOrContext)._conditionalAnd)


		p.SetState(50)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
	    	goto errorExit
	    }
		_la = p.GetTokenStream().LA(1)
	}



errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
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


	// Getter signatures
	AllRelation() []IRelationContext
	Relation(i int) IRelationContext
	AllLOGICAL_AND() []antlr.TerminalNode
	LOGICAL_AND(i int) antlr.TerminalNode

	// IsConditionalAndContext differentiates from other interfaces.
	IsConditionalAndContext()
}

type ConditionalAndContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	e IRelationContext 
	s8 antlr.Token
	ops []antlr.Token
	_relation IRelationContext 
	e1 []IRelationContext
}

func NewEmptyConditionalAndContext() *ConditionalAndContext {
	var p = new(ConditionalAndContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_conditionalAnd
	return p
}

func InitEmptyConditionalAndContext(p *ConditionalAndContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_conditionalAnd
}

func (*ConditionalAndContext) IsConditionalAndContext() {}

func NewConditionalAndContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *ConditionalAndContext {
	var p = new(ConditionalAndContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

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
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IRelationContext); ok {
			len++
		}
	}

	tst := make([]IRelationContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IRelationContext); ok {
			tst[i] = t.(IRelationContext)
			i++
		}
	}

	return tst
}

func (s *ConditionalAndContext) Relation(i int) IRelationContext {
	var t antlr.RuleContext;
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IRelationContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext);
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IRelationContext)
}

func (s *ConditionalAndContext) AllLOGICAL_AND() []antlr.TerminalNode {
	return s.GetTokens(CELParserLOGICAL_AND)
}

func (s *ConditionalAndContext) LOGICAL_AND(i int) antlr.TerminalNode {
	return s.GetToken(CELParserLOGICAL_AND, i)
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

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(51)

		var _x = p.relation(0)

		localctx.(*ConditionalAndContext).e = _x
	}
	p.SetState(56)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_la = p.GetTokenStream().LA(1)


	for _la == CELParserLOGICAL_AND {
		{
			p.SetState(52)

			var _m = p.Match(CELParserLOGICAL_AND)

			localctx.(*ConditionalAndContext).s8 = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}
		localctx.(*ConditionalAndContext).ops = append(localctx.(*ConditionalAndContext).ops, localctx.(*ConditionalAndContext).s8)
		{
			p.SetState(53)

			var _x = p.relation(0)

			localctx.(*ConditionalAndContext)._relation = _x
		}
		localctx.(*ConditionalAndContext).e1 = append(localctx.(*ConditionalAndContext).e1, localctx.(*ConditionalAndContext)._relation)


		p.SetState(58)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
	    	goto errorExit
	    }
		_la = p.GetTokenStream().LA(1)
	}



errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
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


	// Getter signatures
	Calc() ICalcContext
	AllRelation() []IRelationContext
	Relation(i int) IRelationContext
	LESS() antlr.TerminalNode
	LESS_EQUALS() antlr.TerminalNode
	GREATER_EQUALS() antlr.TerminalNode
	GREATER() antlr.TerminalNode
	EQUALS() antlr.TerminalNode
	NOT_EQUALS() antlr.TerminalNode
	IN() antlr.TerminalNode

	// IsRelationContext differentiates from other interfaces.
	IsRelationContext()
}

type RelationContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	op antlr.Token
}

func NewEmptyRelationContext() *RelationContext {
	var p = new(RelationContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_relation
	return p
}

func InitEmptyRelationContext(p *RelationContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_relation
}

func (*RelationContext) IsRelationContext() {}

func NewRelationContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *RelationContext {
	var p = new(RelationContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_relation

	return p
}

func (s *RelationContext) GetParser() antlr.Parser { return s.parser }

func (s *RelationContext) GetOp() antlr.Token { return s.op }


func (s *RelationContext) SetOp(v antlr.Token) { s.op = v }


func (s *RelationContext) Calc() ICalcContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(ICalcContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(ICalcContext)
}

func (s *RelationContext) AllRelation() []IRelationContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IRelationContext); ok {
			len++
		}
	}

	tst := make([]IRelationContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IRelationContext); ok {
			tst[i] = t.(IRelationContext)
			i++
		}
	}

	return tst
}

func (s *RelationContext) Relation(i int) IRelationContext {
	var t antlr.RuleContext;
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IRelationContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext);
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IRelationContext)
}

func (s *RelationContext) LESS() antlr.TerminalNode {
	return s.GetToken(CELParserLESS, 0)
}

func (s *RelationContext) LESS_EQUALS() antlr.TerminalNode {
	return s.GetToken(CELParserLESS_EQUALS, 0)
}

func (s *RelationContext) GREATER_EQUALS() antlr.TerminalNode {
	return s.GetToken(CELParserGREATER_EQUALS, 0)
}

func (s *RelationContext) GREATER() antlr.TerminalNode {
	return s.GetToken(CELParserGREATER, 0)
}

func (s *RelationContext) EQUALS() antlr.TerminalNode {
	return s.GetToken(CELParserEQUALS, 0)
}

func (s *RelationContext) NOT_EQUALS() antlr.TerminalNode {
	return s.GetToken(CELParserNOT_EQUALS, 0)
}

func (s *RelationContext) IN() antlr.TerminalNode {
	return s.GetToken(CELParserIN, 0)
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

	var _alt int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(60)
		p.calc(0)
	}

	p.GetParserRuleContext().SetStop(p.GetTokenStream().LT(-1))
	p.SetState(67)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 3, p.GetParserRuleContext())
	if p.HasError() {
		goto errorExit
	}
	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			if p.GetParseListeners() != nil {
				p.TriggerExitRuleEvent()
			}
			_prevctx = localctx
			localctx = NewRelationContext(p, _parentctx, _parentState)
			p.PushNewRecursionContext(localctx, _startState, CELParserRULE_relation)
			p.SetState(62)

			if !(p.Precpred(p.GetParserRuleContext(), 1)) {
				p.SetError(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 1)", ""))
				goto errorExit
			}
			{
				p.SetState(63)

				var _lt = p.GetTokenStream().LT(1)

				localctx.(*RelationContext).op = _lt

				_la = p.GetTokenStream().LA(1)

				if !(((int64(_la) & ^0x3f) == 0 && ((int64(1) << _la) & 254) != 0)) {
					var _ri = p.GetErrorHandler().RecoverInline(p)

					localctx.(*RelationContext).op = _ri
				} else {
					p.GetErrorHandler().ReportMatch(p)
					p.Consume()
				}
			}
			{
				p.SetState(64)
				p.relation(2)
			}


		}
		p.SetState(69)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
	    	goto errorExit
	    }
		_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 3, p.GetParserRuleContext())
		if p.HasError() {
			goto errorExit
		}
	}



	errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.UnrollRecursionContexts(_parentctx)
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
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


	// Getter signatures
	Unary() IUnaryContext
	AllCalc() []ICalcContext
	Calc(i int) ICalcContext
	STAR() antlr.TerminalNode
	SLASH() antlr.TerminalNode
	PERCENT() antlr.TerminalNode
	PLUS() antlr.TerminalNode
	MINUS() antlr.TerminalNode

	// IsCalcContext differentiates from other interfaces.
	IsCalcContext()
}

type CalcContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	op antlr.Token
}

func NewEmptyCalcContext() *CalcContext {
	var p = new(CalcContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_calc
	return p
}

func InitEmptyCalcContext(p *CalcContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_calc
}

func (*CalcContext) IsCalcContext() {}

func NewCalcContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *CalcContext {
	var p = new(CalcContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_calc

	return p
}

func (s *CalcContext) GetParser() antlr.Parser { return s.parser }

func (s *CalcContext) GetOp() antlr.Token { return s.op }


func (s *CalcContext) SetOp(v antlr.Token) { s.op = v }


func (s *CalcContext) Unary() IUnaryContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IUnaryContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IUnaryContext)
}

func (s *CalcContext) AllCalc() []ICalcContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(ICalcContext); ok {
			len++
		}
	}

	tst := make([]ICalcContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(ICalcContext); ok {
			tst[i] = t.(ICalcContext)
			i++
		}
	}

	return tst
}

func (s *CalcContext) Calc(i int) ICalcContext {
	var t antlr.RuleContext;
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(ICalcContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext);
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(ICalcContext)
}

func (s *CalcContext) STAR() antlr.TerminalNode {
	return s.GetToken(CELParserSTAR, 0)
}

func (s *CalcContext) SLASH() antlr.TerminalNode {
	return s.GetToken(CELParserSLASH, 0)
}

func (s *CalcContext) PERCENT() antlr.TerminalNode {
	return s.GetToken(CELParserPERCENT, 0)
}

func (s *CalcContext) PLUS() antlr.TerminalNode {
	return s.GetToken(CELParserPLUS, 0)
}

func (s *CalcContext) MINUS() antlr.TerminalNode {
	return s.GetToken(CELParserMINUS, 0)
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

	var _alt int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(71)
		p.Unary()
	}

	p.GetParserRuleContext().SetStop(p.GetTokenStream().LT(-1))
	p.SetState(81)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 5, p.GetParserRuleContext())
	if p.HasError() {
		goto errorExit
	}
	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			if p.GetParseListeners() != nil {
				p.TriggerExitRuleEvent()
			}
			_prevctx = localctx
			p.SetState(79)
			p.GetErrorHandler().Sync(p)
			if p.HasError() {
				goto errorExit
			}

			switch p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 4, p.GetParserRuleContext()) {
			case 1:
				localctx = NewCalcContext(p, _parentctx, _parentState)
				p.PushNewRecursionContext(localctx, _startState, CELParserRULE_calc)
				p.SetState(73)

				if !(p.Precpred(p.GetParserRuleContext(), 2)) {
					p.SetError(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 2)", ""))
					goto errorExit
				}
				{
					p.SetState(74)

					var _lt = p.GetTokenStream().LT(1)

					localctx.(*CalcContext).op = _lt

					_la = p.GetTokenStream().LA(1)

					if !(((int64(_la) & ^0x3f) == 0 && ((int64(1) << _la) & 58720256) != 0)) {
						var _ri = p.GetErrorHandler().RecoverInline(p)

						localctx.(*CalcContext).op = _ri
					} else {
						p.GetErrorHandler().ReportMatch(p)
						p.Consume()
					}
				}
				{
					p.SetState(75)
					p.calc(3)
				}


			case 2:
				localctx = NewCalcContext(p, _parentctx, _parentState)
				p.PushNewRecursionContext(localctx, _startState, CELParserRULE_calc)
				p.SetState(76)

				if !(p.Precpred(p.GetParserRuleContext(), 1)) {
					p.SetError(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 1)", ""))
					goto errorExit
				}
				{
					p.SetState(77)

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
				}
				{
					p.SetState(78)
					p.calc(2)
				}

			case antlr.ATNInvalidAltNumber:
				goto errorExit
			}

		}
		p.SetState(83)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
	    	goto errorExit
	    }
		_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 5, p.GetParserRuleContext())
		if p.HasError() {
			goto errorExit
		}
	}



	errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.UnrollRecursionContexts(_parentctx)
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
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
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyUnaryContext() *UnaryContext {
	var p = new(UnaryContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_unary
	return p
}

func InitEmptyUnaryContext(p *UnaryContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_unary
}

func (*UnaryContext) IsUnaryContext() {}

func NewUnaryContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *UnaryContext {
	var p = new(UnaryContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_unary

	return p
}

func (s *UnaryContext) GetParser() antlr.Parser { return s.parser }

func (s *UnaryContext) CopyAll(ctx *UnaryContext) {
	s.CopyFrom(&ctx.BaseParserRuleContext)
}

func (s *UnaryContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *UnaryContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}




type LogicalNotContext struct {
	UnaryContext
	s19 antlr.Token
	ops []antlr.Token
}

func NewLogicalNotContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *LogicalNotContext {
	var p = new(LogicalNotContext)

	InitEmptyUnaryContext(&p.UnaryContext)
	p.parser = parser
	p.CopyAll(ctx.(*UnaryContext))

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
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IMemberContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IMemberContext)
}

func (s *LogicalNotContext) AllEXCLAM() []antlr.TerminalNode {
	return s.GetTokens(CELParserEXCLAM)
}

func (s *LogicalNotContext) EXCLAM(i int) antlr.TerminalNode {
	return s.GetToken(CELParserEXCLAM, i)
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
	UnaryContext
}

func NewMemberExprContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *MemberExprContext {
	var p = new(MemberExprContext)

	InitEmptyUnaryContext(&p.UnaryContext)
	p.parser = parser
	p.CopyAll(ctx.(*UnaryContext))

	return p
}

func (s *MemberExprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *MemberExprContext) Member() IMemberContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IMemberContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

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
	UnaryContext
	s18 antlr.Token
	ops []antlr.Token
}

func NewNegateContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *NegateContext {
	var p = new(NegateContext)

	InitEmptyUnaryContext(&p.UnaryContext)
	p.parser = parser
	p.CopyAll(ctx.(*UnaryContext))

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
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IMemberContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IMemberContext)
}

func (s *NegateContext) AllMINUS() []antlr.TerminalNode {
	return s.GetTokens(CELParserMINUS)
}

func (s *NegateContext) MINUS(i int) antlr.TerminalNode {
	return s.GetToken(CELParserMINUS, i)
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

	var _alt int

	p.SetState(97)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}

	switch p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 8, p.GetParserRuleContext()) {
	case 1:
		localctx = NewMemberExprContext(p, localctx)
		p.EnterOuterAlt(localctx, 1)
		{
			p.SetState(84)
			p.member(0)
		}


	case 2:
		localctx = NewLogicalNotContext(p, localctx)
		p.EnterOuterAlt(localctx, 2)
		p.SetState(86)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		for ok := true; ok; ok = _la == CELParserEXCLAM {
			{
				p.SetState(85)

				var _m = p.Match(CELParserEXCLAM)

				localctx.(*LogicalNotContext).s19 = _m
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}
			localctx.(*LogicalNotContext).ops = append(localctx.(*LogicalNotContext).ops, localctx.(*LogicalNotContext).s19)


			p.SetState(88)
			p.GetErrorHandler().Sync(p)
			if p.HasError() {
		    	goto errorExit
		    }
			_la = p.GetTokenStream().LA(1)
		}
		{
			p.SetState(90)
			p.member(0)
		}


	case 3:
		localctx = NewNegateContext(p, localctx)
		p.EnterOuterAlt(localctx, 3)
		p.SetState(92)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_alt = 1
		for ok := true; ok; ok = _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
			switch _alt {
			case 1:
					{
						p.SetState(91)

						var _m = p.Match(CELParserMINUS)

						localctx.(*NegateContext).s18 = _m
						if p.HasError() {
								// Recognition error - abort rule
								goto errorExit
						}
					}
					localctx.(*NegateContext).ops = append(localctx.(*NegateContext).ops, localctx.(*NegateContext).s18)




			default:
				p.SetError(antlr.NewNoViableAltException(p, nil, nil, nil, nil, nil))
				goto errorExit
			}

			p.SetState(94)
			p.GetErrorHandler().Sync(p)
			_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 7, p.GetParserRuleContext())
			if p.HasError() {
				goto errorExit
			}
		}
		{
			p.SetState(96)
			p.member(0)
		}

	case antlr.ATNInvalidAltNumber:
		goto errorExit
	}


errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
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
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyMemberContext() *MemberContext {
	var p = new(MemberContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_member
	return p
}

func InitEmptyMemberContext(p *MemberContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_member
}

func (*MemberContext) IsMemberContext() {}

func NewMemberContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *MemberContext {
	var p = new(MemberContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_member

	return p
}

func (s *MemberContext) GetParser() antlr.Parser { return s.parser }

func (s *MemberContext) CopyAll(ctx *MemberContext) {
	s.CopyFrom(&ctx.BaseParserRuleContext)
}

func (s *MemberContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *MemberContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}





type MemberCallContext struct {
	MemberContext
	op antlr.Token
	id antlr.Token
	open antlr.Token
	args IExprListContext 
}

func NewMemberCallContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *MemberCallContext {
	var p = new(MemberCallContext)

	InitEmptyMemberContext(&p.MemberContext)
	p.parser = parser
	p.CopyAll(ctx.(*MemberContext))

	return p
}


func (s *MemberCallContext) GetOp() antlr.Token { return s.op }

func (s *MemberCallContext) GetId() antlr.Token { return s.id }

func (s *MemberCallContext) GetOpen() antlr.Token { return s.open }


func (s *MemberCallContext) SetOp(v antlr.Token) { s.op = v }

func (s *MemberCallContext) SetId(v antlr.Token) { s.id = v }

func (s *MemberCallContext) SetOpen(v antlr.Token) { s.open = v }


func (s *MemberCallContext) GetArgs() IExprListContext { return s.args }


func (s *MemberCallContext) SetArgs(v IExprListContext) { s.args = v }

func (s *MemberCallContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *MemberCallContext) Member() IMemberContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IMemberContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IMemberContext)
}

func (s *MemberCallContext) RPAREN() antlr.TerminalNode {
	return s.GetToken(CELParserRPAREN, 0)
}

func (s *MemberCallContext) DOT() antlr.TerminalNode {
	return s.GetToken(CELParserDOT, 0)
}

func (s *MemberCallContext) IDENTIFIER() antlr.TerminalNode {
	return s.GetToken(CELParserIDENTIFIER, 0)
}

func (s *MemberCallContext) LPAREN() antlr.TerminalNode {
	return s.GetToken(CELParserLPAREN, 0)
}

func (s *MemberCallContext) ExprList() IExprListContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IExprListContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IExprListContext)
}


func (s *MemberCallContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterMemberCall(s)
	}
}

func (s *MemberCallContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitMemberCall(s)
	}
}

func (s *MemberCallContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitMemberCall(s)

	default:
		return t.VisitChildren(s)
	}
}


type SelectContext struct {
	MemberContext
	op antlr.Token
	opt antlr.Token
	id antlr.Token
}

func NewSelectContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *SelectContext {
	var p = new(SelectContext)

	InitEmptyMemberContext(&p.MemberContext)
	p.parser = parser
	p.CopyAll(ctx.(*MemberContext))

	return p
}


func (s *SelectContext) GetOp() antlr.Token { return s.op }

func (s *SelectContext) GetOpt() antlr.Token { return s.opt }

func (s *SelectContext) GetId() antlr.Token { return s.id }


func (s *SelectContext) SetOp(v antlr.Token) { s.op = v }

func (s *SelectContext) SetOpt(v antlr.Token) { s.opt = v }

func (s *SelectContext) SetId(v antlr.Token) { s.id = v }

func (s *SelectContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *SelectContext) Member() IMemberContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IMemberContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IMemberContext)
}

func (s *SelectContext) DOT() antlr.TerminalNode {
	return s.GetToken(CELParserDOT, 0)
}

func (s *SelectContext) IDENTIFIER() antlr.TerminalNode {
	return s.GetToken(CELParserIDENTIFIER, 0)
}

func (s *SelectContext) QUESTIONMARK() antlr.TerminalNode {
	return s.GetToken(CELParserQUESTIONMARK, 0)
}


func (s *SelectContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterSelect(s)
	}
}

func (s *SelectContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitSelect(s)
	}
}

func (s *SelectContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitSelect(s)

	default:
		return t.VisitChildren(s)
	}
}


type PrimaryExprContext struct {
	MemberContext
}

func NewPrimaryExprContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *PrimaryExprContext {
	var p = new(PrimaryExprContext)

	InitEmptyMemberContext(&p.MemberContext)
	p.parser = parser
	p.CopyAll(ctx.(*MemberContext))

	return p
}

func (s *PrimaryExprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *PrimaryExprContext) Primary() IPrimaryContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IPrimaryContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

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
	MemberContext
	op antlr.Token
	opt antlr.Token
	index IExprContext 
}

func NewIndexContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *IndexContext {
	var p = new(IndexContext)

	InitEmptyMemberContext(&p.MemberContext)
	p.parser = parser
	p.CopyAll(ctx.(*MemberContext))

	return p
}


func (s *IndexContext) GetOp() antlr.Token { return s.op }

func (s *IndexContext) GetOpt() antlr.Token { return s.opt }


func (s *IndexContext) SetOp(v antlr.Token) { s.op = v }

func (s *IndexContext) SetOpt(v antlr.Token) { s.opt = v }


func (s *IndexContext) GetIndex() IExprContext { return s.index }


func (s *IndexContext) SetIndex(v IExprContext) { s.index = v }

func (s *IndexContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *IndexContext) Member() IMemberContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IMemberContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IMemberContext)
}

func (s *IndexContext) RPRACKET() antlr.TerminalNode {
	return s.GetToken(CELParserRPRACKET, 0)
}

func (s *IndexContext) LBRACKET() antlr.TerminalNode {
	return s.GetToken(CELParserLBRACKET, 0)
}

func (s *IndexContext) Expr() IExprContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IExprContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *IndexContext) QUESTIONMARK() antlr.TerminalNode {
	return s.GetToken(CELParserQUESTIONMARK, 0)
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

	var _alt int

	p.EnterOuterAlt(localctx, 1)
	localctx = NewPrimaryExprContext(p, localctx)
	p.SetParserRuleContext(localctx)
	_prevctx = localctx

	{
		p.SetState(100)
		p.Primary()
	}

	p.GetParserRuleContext().SetStop(p.GetTokenStream().LT(-1))
	p.SetState(126)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 13, p.GetParserRuleContext())
	if p.HasError() {
		goto errorExit
	}
	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			if p.GetParseListeners() != nil {
				p.TriggerExitRuleEvent()
			}
			_prevctx = localctx
			p.SetState(124)
			p.GetErrorHandler().Sync(p)
			if p.HasError() {
				goto errorExit
			}

			switch p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 12, p.GetParserRuleContext()) {
			case 1:
				localctx = NewSelectContext(p, NewMemberContext(p, _parentctx, _parentState))
				p.PushNewRecursionContext(localctx, _startState, CELParserRULE_member)
				p.SetState(102)

				if !(p.Precpred(p.GetParserRuleContext(), 3)) {
					p.SetError(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 3)", ""))
					goto errorExit
				}
				{
					p.SetState(103)

					var _m = p.Match(CELParserDOT)

					localctx.(*SelectContext).op = _m
					if p.HasError() {
							// Recognition error - abort rule
							goto errorExit
					}
				}
				p.SetState(105)
				p.GetErrorHandler().Sync(p)
				if p.HasError() {
					goto errorExit
				}
				_la = p.GetTokenStream().LA(1)


				if _la == CELParserQUESTIONMARK {
					{
						p.SetState(104)

						var _m = p.Match(CELParserQUESTIONMARK)

						localctx.(*SelectContext).opt = _m
						if p.HasError() {
								// Recognition error - abort rule
								goto errorExit
						}
					}

				}
				{
					p.SetState(107)

					var _m = p.Match(CELParserIDENTIFIER)

					localctx.(*SelectContext).id = _m
					if p.HasError() {
							// Recognition error - abort rule
							goto errorExit
					}
				}


			case 2:
				localctx = NewMemberCallContext(p, NewMemberContext(p, _parentctx, _parentState))
				p.PushNewRecursionContext(localctx, _startState, CELParserRULE_member)
				p.SetState(108)

				if !(p.Precpred(p.GetParserRuleContext(), 2)) {
					p.SetError(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 2)", ""))
					goto errorExit
				}
				{
					p.SetState(109)

					var _m = p.Match(CELParserDOT)

					localctx.(*MemberCallContext).op = _m
					if p.HasError() {
							// Recognition error - abort rule
							goto errorExit
					}
				}
				{
					p.SetState(110)

					var _m = p.Match(CELParserIDENTIFIER)

					localctx.(*MemberCallContext).id = _m
					if p.HasError() {
							// Recognition error - abort rule
							goto errorExit
					}
				}
				{
					p.SetState(111)

					var _m = p.Match(CELParserLPAREN)

					localctx.(*MemberCallContext).open = _m
					if p.HasError() {
							// Recognition error - abort rule
							goto errorExit
					}
				}
				p.SetState(113)
				p.GetErrorHandler().Sync(p)
				if p.HasError() {
					goto errorExit
				}
				_la = p.GetTokenStream().LA(1)


				if ((int64(_la) & ^0x3f) == 0 && ((int64(1) << _la) & 135762105344) != 0) {
					{
						p.SetState(112)

						var _x = p.ExprList()


						localctx.(*MemberCallContext).args = _x
					}

				}
				{
					p.SetState(115)
					p.Match(CELParserRPAREN)
					if p.HasError() {
							// Recognition error - abort rule
							goto errorExit
					}
				}


			case 3:
				localctx = NewIndexContext(p, NewMemberContext(p, _parentctx, _parentState))
				p.PushNewRecursionContext(localctx, _startState, CELParserRULE_member)
				p.SetState(116)

				if !(p.Precpred(p.GetParserRuleContext(), 1)) {
					p.SetError(antlr.NewFailedPredicateException(p, "p.Precpred(p.GetParserRuleContext(), 1)", ""))
					goto errorExit
				}
				{
					p.SetState(117)

					var _m = p.Match(CELParserLBRACKET)

					localctx.(*IndexContext).op = _m
					if p.HasError() {
							// Recognition error - abort rule
							goto errorExit
					}
				}
				p.SetState(119)
				p.GetErrorHandler().Sync(p)
				if p.HasError() {
					goto errorExit
				}
				_la = p.GetTokenStream().LA(1)


				if _la == CELParserQUESTIONMARK {
					{
						p.SetState(118)

						var _m = p.Match(CELParserQUESTIONMARK)

						localctx.(*IndexContext).opt = _m
						if p.HasError() {
								// Recognition error - abort rule
								goto errorExit
						}
					}

				}
				{
					p.SetState(121)

					var _x = p.Expr()


					localctx.(*IndexContext).index = _x
				}
				{
					p.SetState(122)
					p.Match(CELParserRPRACKET)
					if p.HasError() {
							// Recognition error - abort rule
							goto errorExit
					}
				}

			case antlr.ATNInvalidAltNumber:
				goto errorExit
			}

		}
		p.SetState(128)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
	    	goto errorExit
	    }
		_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 13, p.GetParserRuleContext())
		if p.HasError() {
			goto errorExit
		}
	}



	errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.UnrollRecursionContexts(_parentctx)
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
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
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyPrimaryContext() *PrimaryContext {
	var p = new(PrimaryContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_primary
	return p
}

func InitEmptyPrimaryContext(p *PrimaryContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_primary
}

func (*PrimaryContext) IsPrimaryContext() {}

func NewPrimaryContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *PrimaryContext {
	var p = new(PrimaryContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_primary

	return p
}

func (s *PrimaryContext) GetParser() antlr.Parser { return s.parser }

func (s *PrimaryContext) CopyAll(ctx *PrimaryContext) {
	s.CopyFrom(&ctx.BaseParserRuleContext)
}

func (s *PrimaryContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *PrimaryContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}




type CreateListContext struct {
	PrimaryContext
	op antlr.Token
	elems IListInitContext 
}

func NewCreateListContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *CreateListContext {
	var p = new(CreateListContext)

	InitEmptyPrimaryContext(&p.PrimaryContext)
	p.parser = parser
	p.CopyAll(ctx.(*PrimaryContext))

	return p
}


func (s *CreateListContext) GetOp() antlr.Token { return s.op }


func (s *CreateListContext) SetOp(v antlr.Token) { s.op = v }


func (s *CreateListContext) GetElems() IListInitContext { return s.elems }


func (s *CreateListContext) SetElems(v IListInitContext) { s.elems = v }

func (s *CreateListContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *CreateListContext) RPRACKET() antlr.TerminalNode {
	return s.GetToken(CELParserRPRACKET, 0)
}

func (s *CreateListContext) LBRACKET() antlr.TerminalNode {
	return s.GetToken(CELParserLBRACKET, 0)
}

func (s *CreateListContext) COMMA() antlr.TerminalNode {
	return s.GetToken(CELParserCOMMA, 0)
}

func (s *CreateListContext) ListInit() IListInitContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IListInitContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IListInitContext)
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
	PrimaryContext
	op antlr.Token
	entries IMapInitializerListContext 
}

func NewCreateStructContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *CreateStructContext {
	var p = new(CreateStructContext)

	InitEmptyPrimaryContext(&p.PrimaryContext)
	p.parser = parser
	p.CopyAll(ctx.(*PrimaryContext))

	return p
}


func (s *CreateStructContext) GetOp() antlr.Token { return s.op }


func (s *CreateStructContext) SetOp(v antlr.Token) { s.op = v }


func (s *CreateStructContext) GetEntries() IMapInitializerListContext { return s.entries }


func (s *CreateStructContext) SetEntries(v IMapInitializerListContext) { s.entries = v }

func (s *CreateStructContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *CreateStructContext) RBRACE() antlr.TerminalNode {
	return s.GetToken(CELParserRBRACE, 0)
}

func (s *CreateStructContext) LBRACE() antlr.TerminalNode {
	return s.GetToken(CELParserLBRACE, 0)
}

func (s *CreateStructContext) COMMA() antlr.TerminalNode {
	return s.GetToken(CELParserCOMMA, 0)
}

func (s *CreateStructContext) MapInitializerList() IMapInitializerListContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IMapInitializerListContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

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
	PrimaryContext
}

func NewConstantLiteralContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *ConstantLiteralContext {
	var p = new(ConstantLiteralContext)

	InitEmptyPrimaryContext(&p.PrimaryContext)
	p.parser = parser
	p.CopyAll(ctx.(*PrimaryContext))

	return p
}

func (s *ConstantLiteralContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *ConstantLiteralContext) Literal() ILiteralContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(ILiteralContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

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
	PrimaryContext
	e IExprContext 
}

func NewNestedContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *NestedContext {
	var p = new(NestedContext)

	InitEmptyPrimaryContext(&p.PrimaryContext)
	p.parser = parser
	p.CopyAll(ctx.(*PrimaryContext))

	return p
}


func (s *NestedContext) GetE() IExprContext { return s.e }


func (s *NestedContext) SetE(v IExprContext) { s.e = v }

func (s *NestedContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *NestedContext) LPAREN() antlr.TerminalNode {
	return s.GetToken(CELParserLPAREN, 0)
}

func (s *NestedContext) RPAREN() antlr.TerminalNode {
	return s.GetToken(CELParserRPAREN, 0)
}

func (s *NestedContext) Expr() IExprContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IExprContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

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


type CreateMessageContext struct {
	PrimaryContext
	leadingDot antlr.Token
	_IDENTIFIER antlr.Token
	ids []antlr.Token
	s16 antlr.Token
	ops []antlr.Token
	op antlr.Token
	entries IFieldInitializerListContext 
}

func NewCreateMessageContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *CreateMessageContext {
	var p = new(CreateMessageContext)

	InitEmptyPrimaryContext(&p.PrimaryContext)
	p.parser = parser
	p.CopyAll(ctx.(*PrimaryContext))

	return p
}


func (s *CreateMessageContext) GetLeadingDot() antlr.Token { return s.leadingDot }

func (s *CreateMessageContext) Get_IDENTIFIER() antlr.Token { return s._IDENTIFIER }

func (s *CreateMessageContext) GetS16() antlr.Token { return s.s16 }

func (s *CreateMessageContext) GetOp() antlr.Token { return s.op }


func (s *CreateMessageContext) SetLeadingDot(v antlr.Token) { s.leadingDot = v }

func (s *CreateMessageContext) Set_IDENTIFIER(v antlr.Token) { s._IDENTIFIER = v }

func (s *CreateMessageContext) SetS16(v antlr.Token) { s.s16 = v }

func (s *CreateMessageContext) SetOp(v antlr.Token) { s.op = v }


func (s *CreateMessageContext) GetIds() []antlr.Token { return s.ids }

func (s *CreateMessageContext) GetOps() []antlr.Token { return s.ops }


func (s *CreateMessageContext) SetIds(v []antlr.Token) { s.ids = v }

func (s *CreateMessageContext) SetOps(v []antlr.Token) { s.ops = v }


func (s *CreateMessageContext) GetEntries() IFieldInitializerListContext { return s.entries }


func (s *CreateMessageContext) SetEntries(v IFieldInitializerListContext) { s.entries = v }

func (s *CreateMessageContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *CreateMessageContext) RBRACE() antlr.TerminalNode {
	return s.GetToken(CELParserRBRACE, 0)
}

func (s *CreateMessageContext) AllIDENTIFIER() []antlr.TerminalNode {
	return s.GetTokens(CELParserIDENTIFIER)
}

func (s *CreateMessageContext) IDENTIFIER(i int) antlr.TerminalNode {
	return s.GetToken(CELParserIDENTIFIER, i)
}

func (s *CreateMessageContext) LBRACE() antlr.TerminalNode {
	return s.GetToken(CELParserLBRACE, 0)
}

func (s *CreateMessageContext) COMMA() antlr.TerminalNode {
	return s.GetToken(CELParserCOMMA, 0)
}

func (s *CreateMessageContext) AllDOT() []antlr.TerminalNode {
	return s.GetTokens(CELParserDOT)
}

func (s *CreateMessageContext) DOT(i int) antlr.TerminalNode {
	return s.GetToken(CELParserDOT, i)
}

func (s *CreateMessageContext) FieldInitializerList() IFieldInitializerListContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IFieldInitializerListContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

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


type IdentOrGlobalCallContext struct {
	PrimaryContext
	leadingDot antlr.Token
	id antlr.Token
	op antlr.Token
	args IExprListContext 
}

func NewIdentOrGlobalCallContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *IdentOrGlobalCallContext {
	var p = new(IdentOrGlobalCallContext)

	InitEmptyPrimaryContext(&p.PrimaryContext)
	p.parser = parser
	p.CopyAll(ctx.(*PrimaryContext))

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

func (s *IdentOrGlobalCallContext) RPAREN() antlr.TerminalNode {
	return s.GetToken(CELParserRPAREN, 0)
}

func (s *IdentOrGlobalCallContext) DOT() antlr.TerminalNode {
	return s.GetToken(CELParserDOT, 0)
}

func (s *IdentOrGlobalCallContext) LPAREN() antlr.TerminalNode {
	return s.GetToken(CELParserLPAREN, 0)
}

func (s *IdentOrGlobalCallContext) ExprList() IExprListContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IExprListContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

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

	p.SetState(180)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}

	switch p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 25, p.GetParserRuleContext()) {
	case 1:
		localctx = NewIdentOrGlobalCallContext(p, localctx)
		p.EnterOuterAlt(localctx, 1)
		p.SetState(130)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		if _la == CELParserDOT {
			{
				p.SetState(129)

				var _m = p.Match(CELParserDOT)

				localctx.(*IdentOrGlobalCallContext).leadingDot = _m
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}

		}
		{
			p.SetState(132)

			var _m = p.Match(CELParserIDENTIFIER)

			localctx.(*IdentOrGlobalCallContext).id = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}
		p.SetState(138)
		p.GetErrorHandler().Sync(p)


		if p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 16, p.GetParserRuleContext()) == 1 {
			{
				p.SetState(133)

				var _m = p.Match(CELParserLPAREN)

				localctx.(*IdentOrGlobalCallContext).op = _m
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}
			p.SetState(135)
			p.GetErrorHandler().Sync(p)
			if p.HasError() {
				goto errorExit
			}
			_la = p.GetTokenStream().LA(1)


			if ((int64(_la) & ^0x3f) == 0 && ((int64(1) << _la) & 135762105344) != 0) {
				{
					p.SetState(134)

					var _x = p.ExprList()


					localctx.(*IdentOrGlobalCallContext).args = _x
				}

			}
			{
				p.SetState(137)
				p.Match(CELParserRPAREN)
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}

			} else if p.HasError() { // JIM
				goto errorExit
		}


	case 2:
		localctx = NewNestedContext(p, localctx)
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(140)
			p.Match(CELParserLPAREN)
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}
		{
			p.SetState(141)

			var _x = p.Expr()


			localctx.(*NestedContext).e = _x
		}
		{
			p.SetState(142)
			p.Match(CELParserRPAREN)
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}


	case 3:
		localctx = NewCreateListContext(p, localctx)
		p.EnterOuterAlt(localctx, 3)
		{
			p.SetState(144)

			var _m = p.Match(CELParserLBRACKET)

			localctx.(*CreateListContext).op = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}
		p.SetState(146)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		if ((int64(_la) & ^0x3f) == 0 && ((int64(1) << _la) & 135763153920) != 0) {
			{
				p.SetState(145)

				var _x = p.ListInit()


				localctx.(*CreateListContext).elems = _x
			}

		}
		p.SetState(149)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		if _la == CELParserCOMMA {
			{
				p.SetState(148)
				p.Match(CELParserCOMMA)
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}

		}
		{
			p.SetState(151)
			p.Match(CELParserRPRACKET)
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}


	case 4:
		localctx = NewCreateStructContext(p, localctx)
		p.EnterOuterAlt(localctx, 4)
		{
			p.SetState(152)

			var _m = p.Match(CELParserLBRACE)

			localctx.(*CreateStructContext).op = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}
		p.SetState(154)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		if ((int64(_la) & ^0x3f) == 0 && ((int64(1) << _la) & 135763153920) != 0) {
			{
				p.SetState(153)

				var _x = p.MapInitializerList()


				localctx.(*CreateStructContext).entries = _x
			}

		}
		p.SetState(157)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		if _la == CELParserCOMMA {
			{
				p.SetState(156)
				p.Match(CELParserCOMMA)
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}

		}
		{
			p.SetState(159)
			p.Match(CELParserRBRACE)
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}


	case 5:
		localctx = NewCreateMessageContext(p, localctx)
		p.EnterOuterAlt(localctx, 5)
		p.SetState(161)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		if _la == CELParserDOT {
			{
				p.SetState(160)

				var _m = p.Match(CELParserDOT)

				localctx.(*CreateMessageContext).leadingDot = _m
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}

		}
		{
			p.SetState(163)

			var _m = p.Match(CELParserIDENTIFIER)

			localctx.(*CreateMessageContext)._IDENTIFIER = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}
		localctx.(*CreateMessageContext).ids = append(localctx.(*CreateMessageContext).ids, localctx.(*CreateMessageContext)._IDENTIFIER)
		p.SetState(168)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		for _la == CELParserDOT {
			{
				p.SetState(164)

				var _m = p.Match(CELParserDOT)

				localctx.(*CreateMessageContext).s16 = _m
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}
			localctx.(*CreateMessageContext).ops = append(localctx.(*CreateMessageContext).ops, localctx.(*CreateMessageContext).s16)
			{
				p.SetState(165)

				var _m = p.Match(CELParserIDENTIFIER)

				localctx.(*CreateMessageContext)._IDENTIFIER = _m
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}
			localctx.(*CreateMessageContext).ids = append(localctx.(*CreateMessageContext).ids, localctx.(*CreateMessageContext)._IDENTIFIER)


			p.SetState(170)
			p.GetErrorHandler().Sync(p)
			if p.HasError() {
		    	goto errorExit
		    }
			_la = p.GetTokenStream().LA(1)
		}
		{
			p.SetState(171)

			var _m = p.Match(CELParserLBRACE)

			localctx.(*CreateMessageContext).op = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}
		p.SetState(173)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		if _la == CELParserQUESTIONMARK || _la == CELParserIDENTIFIER {
			{
				p.SetState(172)

				var _x = p.FieldInitializerList()


				localctx.(*CreateMessageContext).entries = _x
			}

		}
		p.SetState(176)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		if _la == CELParserCOMMA {
			{
				p.SetState(175)
				p.Match(CELParserCOMMA)
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}

		}
		{
			p.SetState(178)
			p.Match(CELParserRBRACE)
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}


	case 6:
		localctx = NewConstantLiteralContext(p, localctx)
		p.EnterOuterAlt(localctx, 6)
		{
			p.SetState(179)
			p.Literal()
		}

	case antlr.ATNInvalidAltNumber:
		goto errorExit
	}


errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
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


	// Getter signatures
	AllExpr() []IExprContext
	Expr(i int) IExprContext
	AllCOMMA() []antlr.TerminalNode
	COMMA(i int) antlr.TerminalNode

	// IsExprListContext differentiates from other interfaces.
	IsExprListContext()
}

type ExprListContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	_expr IExprContext 
	e []IExprContext
}

func NewEmptyExprListContext() *ExprListContext {
	var p = new(ExprListContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_exprList
	return p
}

func InitEmptyExprListContext(p *ExprListContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_exprList
}

func (*ExprListContext) IsExprListContext() {}

func NewExprListContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *ExprListContext {
	var p = new(ExprListContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

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
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IExprContext); ok {
			len++
		}
	}

	tst := make([]IExprContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IExprContext); ok {
			tst[i] = t.(IExprContext)
			i++
		}
	}

	return tst
}

func (s *ExprListContext) Expr(i int) IExprContext {
	var t antlr.RuleContext;
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IExprContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext);
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *ExprListContext) AllCOMMA() []antlr.TerminalNode {
	return s.GetTokens(CELParserCOMMA)
}

func (s *ExprListContext) COMMA(i int) antlr.TerminalNode {
	return s.GetToken(CELParserCOMMA, i)
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
	var _la int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(182)

		var _x = p.Expr()


		localctx.(*ExprListContext)._expr = _x
	}
	localctx.(*ExprListContext).e = append(localctx.(*ExprListContext).e, localctx.(*ExprListContext)._expr)
	p.SetState(187)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_la = p.GetTokenStream().LA(1)


	for _la == CELParserCOMMA {
		{
			p.SetState(183)
			p.Match(CELParserCOMMA)
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}
		{
			p.SetState(184)

			var _x = p.Expr()


			localctx.(*ExprListContext)._expr = _x
		}
		localctx.(*ExprListContext).e = append(localctx.(*ExprListContext).e, localctx.(*ExprListContext)._expr)


		p.SetState(189)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
	    	goto errorExit
	    }
		_la = p.GetTokenStream().LA(1)
	}



errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
}


// IListInitContext is an interface to support dynamic dispatch.
type IListInitContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Get_optExpr returns the _optExpr rule contexts.
	Get_optExpr() IOptExprContext


	// Set_optExpr sets the _optExpr rule contexts.
	Set_optExpr(IOptExprContext)


	// GetElems returns the elems rule context list.
	GetElems() []IOptExprContext


	// SetElems sets the elems rule context list.
	SetElems([]IOptExprContext) 


	// Getter signatures
	AllOptExpr() []IOptExprContext
	OptExpr(i int) IOptExprContext
	AllCOMMA() []antlr.TerminalNode
	COMMA(i int) antlr.TerminalNode

	// IsListInitContext differentiates from other interfaces.
	IsListInitContext()
}

type ListInitContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	_optExpr IOptExprContext 
	elems []IOptExprContext
}

func NewEmptyListInitContext() *ListInitContext {
	var p = new(ListInitContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_listInit
	return p
}

func InitEmptyListInitContext(p *ListInitContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_listInit
}

func (*ListInitContext) IsListInitContext() {}

func NewListInitContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *ListInitContext {
	var p = new(ListInitContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_listInit

	return p
}

func (s *ListInitContext) GetParser() antlr.Parser { return s.parser }

func (s *ListInitContext) Get_optExpr() IOptExprContext { return s._optExpr }


func (s *ListInitContext) Set_optExpr(v IOptExprContext) { s._optExpr = v }


func (s *ListInitContext) GetElems() []IOptExprContext { return s.elems }


func (s *ListInitContext) SetElems(v []IOptExprContext) { s.elems = v }


func (s *ListInitContext) AllOptExpr() []IOptExprContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IOptExprContext); ok {
			len++
		}
	}

	tst := make([]IOptExprContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IOptExprContext); ok {
			tst[i] = t.(IOptExprContext)
			i++
		}
	}

	return tst
}

func (s *ListInitContext) OptExpr(i int) IOptExprContext {
	var t antlr.RuleContext;
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IOptExprContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext);
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IOptExprContext)
}

func (s *ListInitContext) AllCOMMA() []antlr.TerminalNode {
	return s.GetTokens(CELParserCOMMA)
}

func (s *ListInitContext) COMMA(i int) antlr.TerminalNode {
	return s.GetToken(CELParserCOMMA, i)
}

func (s *ListInitContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *ListInitContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}


func (s *ListInitContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterListInit(s)
	}
}

func (s *ListInitContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitListInit(s)
	}
}

func (s *ListInitContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitListInit(s)

	default:
		return t.VisitChildren(s)
	}
}




func (p *CELParser) ListInit() (localctx IListInitContext) {
	localctx = NewListInitContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 20, CELParserRULE_listInit)
	var _alt int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(190)

		var _x = p.OptExpr()


		localctx.(*ListInitContext)._optExpr = _x
	}
	localctx.(*ListInitContext).elems = append(localctx.(*ListInitContext).elems, localctx.(*ListInitContext)._optExpr)
	p.SetState(195)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 27, p.GetParserRuleContext())
	if p.HasError() {
		goto errorExit
	}
	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			{
				p.SetState(191)
				p.Match(CELParserCOMMA)
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}
			{
				p.SetState(192)

				var _x = p.OptExpr()


				localctx.(*ListInitContext)._optExpr = _x
			}
			localctx.(*ListInitContext).elems = append(localctx.(*ListInitContext).elems, localctx.(*ListInitContext)._optExpr)


		}
		p.SetState(197)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
	    	goto errorExit
	    }
		_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 27, p.GetParserRuleContext())
		if p.HasError() {
			goto errorExit
		}
	}



errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
}


// IFieldInitializerListContext is an interface to support dynamic dispatch.
type IFieldInitializerListContext interface {
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


	// Get_optField returns the _optField rule contexts.
	Get_optField() IOptFieldContext

	// Get_expr returns the _expr rule contexts.
	Get_expr() IExprContext


	// Set_optField sets the _optField rule contexts.
	Set_optField(IOptFieldContext)

	// Set_expr sets the _expr rule contexts.
	Set_expr(IExprContext)


	// GetFields returns the fields rule context list.
	GetFields() []IOptFieldContext

	// GetValues returns the values rule context list.
	GetValues() []IExprContext


	// SetFields sets the fields rule context list.
	SetFields([]IOptFieldContext) 

	// SetValues sets the values rule context list.
	SetValues([]IExprContext) 


	// Getter signatures
	AllOptField() []IOptFieldContext
	OptField(i int) IOptFieldContext
	AllCOLON() []antlr.TerminalNode
	COLON(i int) antlr.TerminalNode
	AllExpr() []IExprContext
	Expr(i int) IExprContext
	AllCOMMA() []antlr.TerminalNode
	COMMA(i int) antlr.TerminalNode

	// IsFieldInitializerListContext differentiates from other interfaces.
	IsFieldInitializerListContext()
}

type FieldInitializerListContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	_optField IOptFieldContext 
	fields []IOptFieldContext
	s21 antlr.Token
	cols []antlr.Token
	_expr IExprContext 
	values []IExprContext
}

func NewEmptyFieldInitializerListContext() *FieldInitializerListContext {
	var p = new(FieldInitializerListContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_fieldInitializerList
	return p
}

func InitEmptyFieldInitializerListContext(p *FieldInitializerListContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_fieldInitializerList
}

func (*FieldInitializerListContext) IsFieldInitializerListContext() {}

func NewFieldInitializerListContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *FieldInitializerListContext {
	var p = new(FieldInitializerListContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_fieldInitializerList

	return p
}

func (s *FieldInitializerListContext) GetParser() antlr.Parser { return s.parser }

func (s *FieldInitializerListContext) GetS21() antlr.Token { return s.s21 }


func (s *FieldInitializerListContext) SetS21(v antlr.Token) { s.s21 = v }


func (s *FieldInitializerListContext) GetCols() []antlr.Token { return s.cols }


func (s *FieldInitializerListContext) SetCols(v []antlr.Token) { s.cols = v }


func (s *FieldInitializerListContext) Get_optField() IOptFieldContext { return s._optField }

func (s *FieldInitializerListContext) Get_expr() IExprContext { return s._expr }


func (s *FieldInitializerListContext) Set_optField(v IOptFieldContext) { s._optField = v }

func (s *FieldInitializerListContext) Set_expr(v IExprContext) { s._expr = v }


func (s *FieldInitializerListContext) GetFields() []IOptFieldContext { return s.fields }

func (s *FieldInitializerListContext) GetValues() []IExprContext { return s.values }


func (s *FieldInitializerListContext) SetFields(v []IOptFieldContext) { s.fields = v }

func (s *FieldInitializerListContext) SetValues(v []IExprContext) { s.values = v }


func (s *FieldInitializerListContext) AllOptField() []IOptFieldContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IOptFieldContext); ok {
			len++
		}
	}

	tst := make([]IOptFieldContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IOptFieldContext); ok {
			tst[i] = t.(IOptFieldContext)
			i++
		}
	}

	return tst
}

func (s *FieldInitializerListContext) OptField(i int) IOptFieldContext {
	var t antlr.RuleContext;
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IOptFieldContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext);
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IOptFieldContext)
}

func (s *FieldInitializerListContext) AllCOLON() []antlr.TerminalNode {
	return s.GetTokens(CELParserCOLON)
}

func (s *FieldInitializerListContext) COLON(i int) antlr.TerminalNode {
	return s.GetToken(CELParserCOLON, i)
}

func (s *FieldInitializerListContext) AllExpr() []IExprContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IExprContext); ok {
			len++
		}
	}

	tst := make([]IExprContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IExprContext); ok {
			tst[i] = t.(IExprContext)
			i++
		}
	}

	return tst
}

func (s *FieldInitializerListContext) Expr(i int) IExprContext {
	var t antlr.RuleContext;
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IExprContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext);
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *FieldInitializerListContext) AllCOMMA() []antlr.TerminalNode {
	return s.GetTokens(CELParserCOMMA)
}

func (s *FieldInitializerListContext) COMMA(i int) antlr.TerminalNode {
	return s.GetToken(CELParserCOMMA, i)
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
	p.EnterRule(localctx, 22, CELParserRULE_fieldInitializerList)
	var _alt int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(198)

		var _x = p.OptField()


		localctx.(*FieldInitializerListContext)._optField = _x
	}
	localctx.(*FieldInitializerListContext).fields = append(localctx.(*FieldInitializerListContext).fields, localctx.(*FieldInitializerListContext)._optField)
	{
		p.SetState(199)

		var _m = p.Match(CELParserCOLON)

		localctx.(*FieldInitializerListContext).s21 = _m
		if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
		}
	}
	localctx.(*FieldInitializerListContext).cols = append(localctx.(*FieldInitializerListContext).cols, localctx.(*FieldInitializerListContext).s21)
	{
		p.SetState(200)

		var _x = p.Expr()


		localctx.(*FieldInitializerListContext)._expr = _x
	}
	localctx.(*FieldInitializerListContext).values = append(localctx.(*FieldInitializerListContext).values, localctx.(*FieldInitializerListContext)._expr)
	p.SetState(208)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 28, p.GetParserRuleContext())
	if p.HasError() {
		goto errorExit
	}
	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			{
				p.SetState(201)
				p.Match(CELParserCOMMA)
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}
			{
				p.SetState(202)

				var _x = p.OptField()


				localctx.(*FieldInitializerListContext)._optField = _x
			}
			localctx.(*FieldInitializerListContext).fields = append(localctx.(*FieldInitializerListContext).fields, localctx.(*FieldInitializerListContext)._optField)
			{
				p.SetState(203)

				var _m = p.Match(CELParserCOLON)

				localctx.(*FieldInitializerListContext).s21 = _m
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}
			localctx.(*FieldInitializerListContext).cols = append(localctx.(*FieldInitializerListContext).cols, localctx.(*FieldInitializerListContext).s21)
			{
				p.SetState(204)

				var _x = p.Expr()


				localctx.(*FieldInitializerListContext)._expr = _x
			}
			localctx.(*FieldInitializerListContext).values = append(localctx.(*FieldInitializerListContext).values, localctx.(*FieldInitializerListContext)._expr)


		}
		p.SetState(210)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
	    	goto errorExit
	    }
		_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 28, p.GetParserRuleContext())
		if p.HasError() {
			goto errorExit
		}
	}



errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
}


// IOptFieldContext is an interface to support dynamic dispatch.
type IOptFieldContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// GetOpt returns the opt token.
	GetOpt() antlr.Token 


	// SetOpt sets the opt token.
	SetOpt(antlr.Token) 


	// Getter signatures
	IDENTIFIER() antlr.TerminalNode
	QUESTIONMARK() antlr.TerminalNode

	// IsOptFieldContext differentiates from other interfaces.
	IsOptFieldContext()
}

type OptFieldContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	opt antlr.Token
}

func NewEmptyOptFieldContext() *OptFieldContext {
	var p = new(OptFieldContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_optField
	return p
}

func InitEmptyOptFieldContext(p *OptFieldContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_optField
}

func (*OptFieldContext) IsOptFieldContext() {}

func NewOptFieldContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *OptFieldContext {
	var p = new(OptFieldContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_optField

	return p
}

func (s *OptFieldContext) GetParser() antlr.Parser { return s.parser }

func (s *OptFieldContext) GetOpt() antlr.Token { return s.opt }


func (s *OptFieldContext) SetOpt(v antlr.Token) { s.opt = v }


func (s *OptFieldContext) IDENTIFIER() antlr.TerminalNode {
	return s.GetToken(CELParserIDENTIFIER, 0)
}

func (s *OptFieldContext) QUESTIONMARK() antlr.TerminalNode {
	return s.GetToken(CELParserQUESTIONMARK, 0)
}

func (s *OptFieldContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *OptFieldContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}


func (s *OptFieldContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterOptField(s)
	}
}

func (s *OptFieldContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitOptField(s)
	}
}

func (s *OptFieldContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitOptField(s)

	default:
		return t.VisitChildren(s)
	}
}




func (p *CELParser) OptField() (localctx IOptFieldContext) {
	localctx = NewOptFieldContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 24, CELParserRULE_optField)
	var _la int

	p.EnterOuterAlt(localctx, 1)
	p.SetState(212)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_la = p.GetTokenStream().LA(1)


	if _la == CELParserQUESTIONMARK {
		{
			p.SetState(211)

			var _m = p.Match(CELParserQUESTIONMARK)

			localctx.(*OptFieldContext).opt = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}

	}
	{
		p.SetState(214)
		p.Match(CELParserIDENTIFIER)
		if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
		}
	}



errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
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


	// Get_optExpr returns the _optExpr rule contexts.
	Get_optExpr() IOptExprContext

	// Get_expr returns the _expr rule contexts.
	Get_expr() IExprContext


	// Set_optExpr sets the _optExpr rule contexts.
	Set_optExpr(IOptExprContext)

	// Set_expr sets the _expr rule contexts.
	Set_expr(IExprContext)


	// GetKeys returns the keys rule context list.
	GetKeys() []IOptExprContext

	// GetValues returns the values rule context list.
	GetValues() []IExprContext


	// SetKeys sets the keys rule context list.
	SetKeys([]IOptExprContext) 

	// SetValues sets the values rule context list.
	SetValues([]IExprContext) 


	// Getter signatures
	AllOptExpr() []IOptExprContext
	OptExpr(i int) IOptExprContext
	AllCOLON() []antlr.TerminalNode
	COLON(i int) antlr.TerminalNode
	AllExpr() []IExprContext
	Expr(i int) IExprContext
	AllCOMMA() []antlr.TerminalNode
	COMMA(i int) antlr.TerminalNode

	// IsMapInitializerListContext differentiates from other interfaces.
	IsMapInitializerListContext()
}

type MapInitializerListContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	_optExpr IOptExprContext 
	keys []IOptExprContext
	s21 antlr.Token
	cols []antlr.Token
	_expr IExprContext 
	values []IExprContext
}

func NewEmptyMapInitializerListContext() *MapInitializerListContext {
	var p = new(MapInitializerListContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_mapInitializerList
	return p
}

func InitEmptyMapInitializerListContext(p *MapInitializerListContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_mapInitializerList
}

func (*MapInitializerListContext) IsMapInitializerListContext() {}

func NewMapInitializerListContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *MapInitializerListContext {
	var p = new(MapInitializerListContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_mapInitializerList

	return p
}

func (s *MapInitializerListContext) GetParser() antlr.Parser { return s.parser }

func (s *MapInitializerListContext) GetS21() antlr.Token { return s.s21 }


func (s *MapInitializerListContext) SetS21(v antlr.Token) { s.s21 = v }


func (s *MapInitializerListContext) GetCols() []antlr.Token { return s.cols }


func (s *MapInitializerListContext) SetCols(v []antlr.Token) { s.cols = v }


func (s *MapInitializerListContext) Get_optExpr() IOptExprContext { return s._optExpr }

func (s *MapInitializerListContext) Get_expr() IExprContext { return s._expr }


func (s *MapInitializerListContext) Set_optExpr(v IOptExprContext) { s._optExpr = v }

func (s *MapInitializerListContext) Set_expr(v IExprContext) { s._expr = v }


func (s *MapInitializerListContext) GetKeys() []IOptExprContext { return s.keys }

func (s *MapInitializerListContext) GetValues() []IExprContext { return s.values }


func (s *MapInitializerListContext) SetKeys(v []IOptExprContext) { s.keys = v }

func (s *MapInitializerListContext) SetValues(v []IExprContext) { s.values = v }


func (s *MapInitializerListContext) AllOptExpr() []IOptExprContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IOptExprContext); ok {
			len++
		}
	}

	tst := make([]IOptExprContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IOptExprContext); ok {
			tst[i] = t.(IOptExprContext)
			i++
		}
	}

	return tst
}

func (s *MapInitializerListContext) OptExpr(i int) IOptExprContext {
	var t antlr.RuleContext;
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IOptExprContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext);
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IOptExprContext)
}

func (s *MapInitializerListContext) AllCOLON() []antlr.TerminalNode {
	return s.GetTokens(CELParserCOLON)
}

func (s *MapInitializerListContext) COLON(i int) antlr.TerminalNode {
	return s.GetToken(CELParserCOLON, i)
}

func (s *MapInitializerListContext) AllExpr() []IExprContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IExprContext); ok {
			len++
		}
	}

	tst := make([]IExprContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IExprContext); ok {
			tst[i] = t.(IExprContext)
			i++
		}
	}

	return tst
}

func (s *MapInitializerListContext) Expr(i int) IExprContext {
	var t antlr.RuleContext;
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IExprContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext);
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *MapInitializerListContext) AllCOMMA() []antlr.TerminalNode {
	return s.GetTokens(CELParserCOMMA)
}

func (s *MapInitializerListContext) COMMA(i int) antlr.TerminalNode {
	return s.GetToken(CELParserCOMMA, i)
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
	p.EnterRule(localctx, 26, CELParserRULE_mapInitializerList)
	var _alt int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(216)

		var _x = p.OptExpr()


		localctx.(*MapInitializerListContext)._optExpr = _x
	}
	localctx.(*MapInitializerListContext).keys = append(localctx.(*MapInitializerListContext).keys, localctx.(*MapInitializerListContext)._optExpr)
	{
		p.SetState(217)

		var _m = p.Match(CELParserCOLON)

		localctx.(*MapInitializerListContext).s21 = _m
		if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
		}
	}
	localctx.(*MapInitializerListContext).cols = append(localctx.(*MapInitializerListContext).cols, localctx.(*MapInitializerListContext).s21)
	{
		p.SetState(218)

		var _x = p.Expr()


		localctx.(*MapInitializerListContext)._expr = _x
	}
	localctx.(*MapInitializerListContext).values = append(localctx.(*MapInitializerListContext).values, localctx.(*MapInitializerListContext)._expr)
	p.SetState(226)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 30, p.GetParserRuleContext())
	if p.HasError() {
		goto errorExit
	}
	for _alt != 2 && _alt != antlr.ATNInvalidAltNumber {
		if _alt == 1 {
			{
				p.SetState(219)
				p.Match(CELParserCOMMA)
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}
			{
				p.SetState(220)

				var _x = p.OptExpr()


				localctx.(*MapInitializerListContext)._optExpr = _x
			}
			localctx.(*MapInitializerListContext).keys = append(localctx.(*MapInitializerListContext).keys, localctx.(*MapInitializerListContext)._optExpr)
			{
				p.SetState(221)

				var _m = p.Match(CELParserCOLON)

				localctx.(*MapInitializerListContext).s21 = _m
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}
			localctx.(*MapInitializerListContext).cols = append(localctx.(*MapInitializerListContext).cols, localctx.(*MapInitializerListContext).s21)
			{
				p.SetState(222)

				var _x = p.Expr()


				localctx.(*MapInitializerListContext)._expr = _x
			}
			localctx.(*MapInitializerListContext).values = append(localctx.(*MapInitializerListContext).values, localctx.(*MapInitializerListContext)._expr)


		}
		p.SetState(228)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
	    	goto errorExit
	    }
		_alt = p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 30, p.GetParserRuleContext())
		if p.HasError() {
			goto errorExit
		}
	}



errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
}


// IOptExprContext is an interface to support dynamic dispatch.
type IOptExprContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// GetOpt returns the opt token.
	GetOpt() antlr.Token 


	// SetOpt sets the opt token.
	SetOpt(antlr.Token) 


	// GetE returns the e rule contexts.
	GetE() IExprContext


	// SetE sets the e rule contexts.
	SetE(IExprContext)


	// Getter signatures
	Expr() IExprContext
	QUESTIONMARK() antlr.TerminalNode

	// IsOptExprContext differentiates from other interfaces.
	IsOptExprContext()
}

type OptExprContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
	opt antlr.Token
	e IExprContext 
}

func NewEmptyOptExprContext() *OptExprContext {
	var p = new(OptExprContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_optExpr
	return p
}

func InitEmptyOptExprContext(p *OptExprContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_optExpr
}

func (*OptExprContext) IsOptExprContext() {}

func NewOptExprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *OptExprContext {
	var p = new(OptExprContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_optExpr

	return p
}

func (s *OptExprContext) GetParser() antlr.Parser { return s.parser }

func (s *OptExprContext) GetOpt() antlr.Token { return s.opt }


func (s *OptExprContext) SetOpt(v antlr.Token) { s.opt = v }


func (s *OptExprContext) GetE() IExprContext { return s.e }


func (s *OptExprContext) SetE(v IExprContext) { s.e = v }


func (s *OptExprContext) Expr() IExprContext {
	var t antlr.RuleContext;
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IExprContext); ok {
			t = ctx.(antlr.RuleContext);
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IExprContext)
}

func (s *OptExprContext) QUESTIONMARK() antlr.TerminalNode {
	return s.GetToken(CELParserQUESTIONMARK, 0)
}

func (s *OptExprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *OptExprContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}


func (s *OptExprContext) EnterRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.EnterOptExpr(s)
	}
}

func (s *OptExprContext) ExitRule(listener antlr.ParseTreeListener) {
	if listenerT, ok := listener.(CELListener); ok {
		listenerT.ExitOptExpr(s)
	}
}

func (s *OptExprContext) Accept(visitor antlr.ParseTreeVisitor) interface{} {
	switch t := visitor.(type) {
	case CELVisitor:
		return t.VisitOptExpr(s)

	default:
		return t.VisitChildren(s)
	}
}




func (p *CELParser) OptExpr() (localctx IOptExprContext) {
	localctx = NewOptExprContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 28, CELParserRULE_optExpr)
	var _la int

	p.EnterOuterAlt(localctx, 1)
	p.SetState(230)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_la = p.GetTokenStream().LA(1)


	if _la == CELParserQUESTIONMARK {
		{
			p.SetState(229)

			var _m = p.Match(CELParserQUESTIONMARK)

			localctx.(*OptExprContext).opt = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}

	}
	{
		p.SetState(232)

		var _x = p.Expr()


		localctx.(*OptExprContext).e = _x
	}



errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
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
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyLiteralContext() *LiteralContext {
	var p = new(LiteralContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_literal
	return p
}

func InitEmptyLiteralContext(p *LiteralContext)  {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = CELParserRULE_literal
}

func (*LiteralContext) IsLiteralContext() {}

func NewLiteralContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *LiteralContext {
	var p = new(LiteralContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = CELParserRULE_literal

	return p
}

func (s *LiteralContext) GetParser() antlr.Parser { return s.parser }

func (s *LiteralContext) CopyAll(ctx *LiteralContext) {
	s.CopyFrom(&ctx.BaseParserRuleContext)
}

func (s *LiteralContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *LiteralContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}




type BytesContext struct {
	LiteralContext
	tok antlr.Token
}

func NewBytesContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *BytesContext {
	var p = new(BytesContext)

	InitEmptyLiteralContext(&p.LiteralContext)
	p.parser = parser
	p.CopyAll(ctx.(*LiteralContext))

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
	LiteralContext
	tok antlr.Token
}

func NewUintContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *UintContext {
	var p = new(UintContext)

	InitEmptyLiteralContext(&p.LiteralContext)
	p.parser = parser
	p.CopyAll(ctx.(*LiteralContext))

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
	LiteralContext
	tok antlr.Token
}

func NewNullContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *NullContext {
	var p = new(NullContext)

	InitEmptyLiteralContext(&p.LiteralContext)
	p.parser = parser
	p.CopyAll(ctx.(*LiteralContext))

	return p
}


func (s *NullContext) GetTok() antlr.Token { return s.tok }


func (s *NullContext) SetTok(v antlr.Token) { s.tok = v }

func (s *NullContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *NullContext) NUL() antlr.TerminalNode {
	return s.GetToken(CELParserNUL, 0)
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
	LiteralContext
	tok antlr.Token
}

func NewBoolFalseContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *BoolFalseContext {
	var p = new(BoolFalseContext)

	InitEmptyLiteralContext(&p.LiteralContext)
	p.parser = parser
	p.CopyAll(ctx.(*LiteralContext))

	return p
}


func (s *BoolFalseContext) GetTok() antlr.Token { return s.tok }


func (s *BoolFalseContext) SetTok(v antlr.Token) { s.tok = v }

func (s *BoolFalseContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *BoolFalseContext) CEL_FALSE() antlr.TerminalNode {
	return s.GetToken(CELParserCEL_FALSE, 0)
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
	LiteralContext
	tok antlr.Token
}

func NewStringContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *StringContext {
	var p = new(StringContext)

	InitEmptyLiteralContext(&p.LiteralContext)
	p.parser = parser
	p.CopyAll(ctx.(*LiteralContext))

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
	LiteralContext
	sign antlr.Token
	tok antlr.Token
}

func NewDoubleContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *DoubleContext {
	var p = new(DoubleContext)

	InitEmptyLiteralContext(&p.LiteralContext)
	p.parser = parser
	p.CopyAll(ctx.(*LiteralContext))

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
	LiteralContext
	tok antlr.Token
}

func NewBoolTrueContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *BoolTrueContext {
	var p = new(BoolTrueContext)

	InitEmptyLiteralContext(&p.LiteralContext)
	p.parser = parser
	p.CopyAll(ctx.(*LiteralContext))

	return p
}


func (s *BoolTrueContext) GetTok() antlr.Token { return s.tok }


func (s *BoolTrueContext) SetTok(v antlr.Token) { s.tok = v }

func (s *BoolTrueContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *BoolTrueContext) CEL_TRUE() antlr.TerminalNode {
	return s.GetToken(CELParserCEL_TRUE, 0)
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
	LiteralContext
	sign antlr.Token
	tok antlr.Token
}

func NewIntContext(parser antlr.Parser, ctx antlr.ParserRuleContext) *IntContext {
	var p = new(IntContext)

	InitEmptyLiteralContext(&p.LiteralContext)
	p.parser = parser
	p.CopyAll(ctx.(*LiteralContext))

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
	p.EnterRule(localctx, 30, CELParserRULE_literal)
	var _la int

	p.SetState(248)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}

	switch p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 34, p.GetParserRuleContext()) {
	case 1:
		localctx = NewIntContext(p, localctx)
		p.EnterOuterAlt(localctx, 1)
		p.SetState(235)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		if _la == CELParserMINUS {
			{
				p.SetState(234)

				var _m = p.Match(CELParserMINUS)

				localctx.(*IntContext).sign = _m
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}

		}
		{
			p.SetState(237)

			var _m = p.Match(CELParserNUM_INT)

			localctx.(*IntContext).tok = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}


	case 2:
		localctx = NewUintContext(p, localctx)
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(238)

			var _m = p.Match(CELParserNUM_UINT)

			localctx.(*UintContext).tok = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}


	case 3:
		localctx = NewDoubleContext(p, localctx)
		p.EnterOuterAlt(localctx, 3)
		p.SetState(240)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)


		if _la == CELParserMINUS {
			{
				p.SetState(239)

				var _m = p.Match(CELParserMINUS)

				localctx.(*DoubleContext).sign = _m
				if p.HasError() {
						// Recognition error - abort rule
						goto errorExit
				}
			}

		}
		{
			p.SetState(242)

			var _m = p.Match(CELParserNUM_FLOAT)

			localctx.(*DoubleContext).tok = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}


	case 4:
		localctx = NewStringContext(p, localctx)
		p.EnterOuterAlt(localctx, 4)
		{
			p.SetState(243)

			var _m = p.Match(CELParserSTRING)

			localctx.(*StringContext).tok = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}


	case 5:
		localctx = NewBytesContext(p, localctx)
		p.EnterOuterAlt(localctx, 5)
		{
			p.SetState(244)

			var _m = p.Match(CELParserBYTES)

			localctx.(*BytesContext).tok = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}


	case 6:
		localctx = NewBoolTrueContext(p, localctx)
		p.EnterOuterAlt(localctx, 6)
		{
			p.SetState(245)

			var _m = p.Match(CELParserCEL_TRUE)

			localctx.(*BoolTrueContext).tok = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}


	case 7:
		localctx = NewBoolFalseContext(p, localctx)
		p.EnterOuterAlt(localctx, 7)
		{
			p.SetState(246)

			var _m = p.Match(CELParserCEL_FALSE)

			localctx.(*BoolFalseContext).tok = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}


	case 8:
		localctx = NewNullContext(p, localctx)
		p.EnterOuterAlt(localctx, 8)
		{
			p.SetState(247)

			var _m = p.Match(CELParserNUL)

			localctx.(*NullContext).tok = _m
			if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
			}
		}

	case antlr.ATNInvalidAltNumber:
		goto errorExit
	}


errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
	goto errorExit // Trick to prevent compiler error if the label is not used
}


func (p *CELParser) Sempred(localctx antlr.RuleContext, ruleIndex, predIndex int) bool {
	switch ruleIndex {
	case 4:
			var t *RelationContext = nil
			if localctx != nil { t = localctx.(*RelationContext) }
			return p.Relation_Sempred(t, predIndex)

	case 5:
			var t *CalcContext = nil
			if localctx != nil { t = localctx.(*CalcContext) }
			return p.Calc_Sempred(t, predIndex)

	case 7:
			var t *MemberContext = nil
			if localctx != nil { t = localctx.(*MemberContext) }
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

