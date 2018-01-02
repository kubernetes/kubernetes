package jmespath

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

var lexingTests = []struct {
	expression string
	expected   []token
}{
	{"*", []token{{tStar, "*", 0, 1}}},
	{".", []token{{tDot, ".", 0, 1}}},
	{"[?", []token{{tFilter, "[?", 0, 2}}},
	{"[]", []token{{tFlatten, "[]", 0, 2}}},
	{"(", []token{{tLparen, "(", 0, 1}}},
	{")", []token{{tRparen, ")", 0, 1}}},
	{"[", []token{{tLbracket, "[", 0, 1}}},
	{"]", []token{{tRbracket, "]", 0, 1}}},
	{"{", []token{{tLbrace, "{", 0, 1}}},
	{"}", []token{{tRbrace, "}", 0, 1}}},
	{"||", []token{{tOr, "||", 0, 2}}},
	{"|", []token{{tPipe, "|", 0, 1}}},
	{"29", []token{{tNumber, "29", 0, 2}}},
	{"2", []token{{tNumber, "2", 0, 1}}},
	{"0", []token{{tNumber, "0", 0, 1}}},
	{"-20", []token{{tNumber, "-20", 0, 3}}},
	{"foo", []token{{tUnquotedIdentifier, "foo", 0, 3}}},
	{`"bar"`, []token{{tQuotedIdentifier, "bar", 0, 3}}},
	// Escaping the delimiter
	{`"bar\"baz"`, []token{{tQuotedIdentifier, `bar"baz`, 0, 7}}},
	{",", []token{{tComma, ",", 0, 1}}},
	{":", []token{{tColon, ":", 0, 1}}},
	{"<", []token{{tLT, "<", 0, 1}}},
	{"<=", []token{{tLTE, "<=", 0, 2}}},
	{">", []token{{tGT, ">", 0, 1}}},
	{">=", []token{{tGTE, ">=", 0, 2}}},
	{"==", []token{{tEQ, "==", 0, 2}}},
	{"!=", []token{{tNE, "!=", 0, 2}}},
	{"`[0, 1, 2]`", []token{{tJSONLiteral, "[0, 1, 2]", 1, 9}}},
	{"'foo'", []token{{tStringLiteral, "foo", 1, 3}}},
	{"'a'", []token{{tStringLiteral, "a", 1, 1}}},
	{`'foo\'bar'`, []token{{tStringLiteral, "foo'bar", 1, 7}}},
	{"@", []token{{tCurrent, "@", 0, 1}}},
	{"&", []token{{tExpref, "&", 0, 1}}},
	// Quoted identifier unicode escape sequences
	{`"\u2713"`, []token{{tQuotedIdentifier, "✓", 0, 3}}},
	{`"\\"`, []token{{tQuotedIdentifier, `\`, 0, 1}}},
	{"`\"foo\"`", []token{{tJSONLiteral, "\"foo\"", 1, 5}}},
	// Combinations of tokens.
	{"foo.bar", []token{
		{tUnquotedIdentifier, "foo", 0, 3},
		{tDot, ".", 3, 1},
		{tUnquotedIdentifier, "bar", 4, 3},
	}},
	{"foo[0]", []token{
		{tUnquotedIdentifier, "foo", 0, 3},
		{tLbracket, "[", 3, 1},
		{tNumber, "0", 4, 1},
		{tRbracket, "]", 5, 1},
	}},
	{"foo[?a<b]", []token{
		{tUnquotedIdentifier, "foo", 0, 3},
		{tFilter, "[?", 3, 2},
		{tUnquotedIdentifier, "a", 5, 1},
		{tLT, "<", 6, 1},
		{tUnquotedIdentifier, "b", 7, 1},
		{tRbracket, "]", 8, 1},
	}},
}

func TestCanLexTokens(t *testing.T) {
	assert := assert.New(t)
	lexer := NewLexer()
	for _, tt := range lexingTests {
		tokens, err := lexer.tokenize(tt.expression)
		if assert.Nil(err) {
			errMsg := fmt.Sprintf("Mismatch expected number of tokens: (expected: %s, actual: %s)",
				tt.expected, tokens)
			tt.expected = append(tt.expected, token{tEOF, "", len(tt.expression), 0})
			if assert.Equal(len(tt.expected), len(tokens), errMsg) {
				for i, token := range tokens {
					expected := tt.expected[i]
					assert.Equal(expected, token, "Token not equal")
				}
			}
		}
	}
}

var lexingErrorTests = []struct {
	expression string
	msg        string
}{
	{"'foo", "Missing closing single quote"},
	{"[?foo==bar?]", "Unknown char '?'"},
}

func TestLexingErrors(t *testing.T) {
	assert := assert.New(t)
	lexer := NewLexer()
	for _, tt := range lexingErrorTests {
		_, err := lexer.tokenize(tt.expression)
		assert.NotNil(err, fmt.Sprintf("Expected lexing error: %s", tt.msg))
	}
}

var exprIdentifier = "abcdefghijklmnopqrstuvwxyz"
var exprSubexpr = "abcdefghijklmnopqrstuvwxyz.abcdefghijklmnopqrstuvwxyz"
var deeplyNested50 = "j49.j48.j47.j46.j45.j44.j43.j42.j41.j40.j39.j38.j37.j36.j35.j34.j33.j32.j31.j30.j29.j28.j27.j26.j25.j24.j23.j22.j21.j20.j19.j18.j17.j16.j15.j14.j13.j12.j11.j10.j9.j8.j7.j6.j5.j4.j3.j2.j1.j0"
var deeplyNested50Pipe = "j49|j48|j47|j46|j45|j44|j43|j42|j41|j40|j39|j38|j37|j36|j35|j34|j33|j32|j31|j30|j29|j28|j27|j26|j25|j24|j23|j22|j21|j20|j19|j18|j17|j16|j15|j14|j13|j12|j11|j10|j9|j8|j7|j6|j5|j4|j3|j2|j1|j0"
var deeplyNested50Index = "[49][48][47][46][45][44][43][42][41][40][39][38][37][36][35][34][33][32][31][30][29][28][27][26][25][24][23][22][21][20][19][18][17][16][15][14][13][12][11][10][9][8][7][6][5][4][3][2][1][0]"
var deepProjection104 = "a[*].b[*].c[*].d[*].e[*].f[*].g[*].h[*].i[*].j[*].k[*].l[*].m[*].n[*].o[*].p[*].q[*].r[*].s[*].t[*].u[*].v[*].w[*].x[*].y[*].z[*].a[*].b[*].c[*].d[*].e[*].f[*].g[*].h[*].i[*].j[*].k[*].l[*].m[*].n[*].o[*].p[*].q[*].r[*].s[*].t[*].u[*].v[*].w[*].x[*].y[*].z[*].a[*].b[*].c[*].d[*].e[*].f[*].g[*].h[*].i[*].j[*].k[*].l[*].m[*].n[*].o[*].p[*].q[*].r[*].s[*].t[*].u[*].v[*].w[*].x[*].y[*].z[*].a[*].b[*].c[*].d[*].e[*].f[*].g[*].h[*].i[*].j[*].k[*].l[*].m[*].n[*].o[*].p[*].q[*].r[*].s[*].t[*].u[*].v[*].w[*].x[*].y[*].z[*]"
var exprQuotedIdentifier = `"abcdefghijklmnopqrstuvwxyz.abcdefghijklmnopqrstuvwxyz"`
var quotedIdentifierEscapes = `"\n\r\b\t\n\r\b\t\n\r\b\t\n\r\b\t\n\r\b\t\n\r\b\t\n\r\b\t"`
var rawStringLiteral = `'abcdefghijklmnopqrstuvwxyz.abcdefghijklmnopqrstuvwxyz'`

func BenchmarkLexIdentifier(b *testing.B) {
	runLexBenchmark(b, exprIdentifier)
}

func BenchmarkLexSubexpression(b *testing.B) {
	runLexBenchmark(b, exprSubexpr)
}

func BenchmarkLexDeeplyNested50(b *testing.B) {
	runLexBenchmark(b, deeplyNested50)
}

func BenchmarkLexDeepNested50Pipe(b *testing.B) {
	runLexBenchmark(b, deeplyNested50Pipe)
}

func BenchmarkLexDeepNested50Index(b *testing.B) {
	runLexBenchmark(b, deeplyNested50Index)
}

func BenchmarkLexQuotedIdentifier(b *testing.B) {
	runLexBenchmark(b, exprQuotedIdentifier)
}

func BenchmarkLexQuotedIdentifierEscapes(b *testing.B) {
	runLexBenchmark(b, quotedIdentifierEscapes)
}

func BenchmarkLexRawStringLiteral(b *testing.B) {
	runLexBenchmark(b, rawStringLiteral)
}

func BenchmarkLexDeepProjection104(b *testing.B) {
	runLexBenchmark(b, deepProjection104)
}

func runLexBenchmark(b *testing.B, expression string) {
	lexer := NewLexer()
	for i := 0; i < b.N; i++ {
		lexer.tokenize(expression)
	}
}
