package influxql_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/influxdata/influxdb/influxql"
)

// Ensure the scanner can scan tokens correctly.
func TestScanner_Scan(t *testing.T) {
	var tests = []struct {
		s   string
		tok influxql.Token
		lit string
		pos influxql.Pos
	}{
		// Special tokens (EOF, ILLEGAL, WS)
		{s: ``, tok: influxql.EOF},
		{s: `#`, tok: influxql.ILLEGAL, lit: `#`},
		{s: ` `, tok: influxql.WS, lit: " "},
		{s: "\t", tok: influxql.WS, lit: "\t"},
		{s: "\n", tok: influxql.WS, lit: "\n"},
		{s: "\r", tok: influxql.WS, lit: "\n"},
		{s: "\r\n", tok: influxql.WS, lit: "\n"},
		{s: "\rX", tok: influxql.WS, lit: "\n"},
		{s: "\n\r", tok: influxql.WS, lit: "\n\n"},
		{s: " \n\t \r\n\t", tok: influxql.WS, lit: " \n\t \n\t"},
		{s: " foo", tok: influxql.WS, lit: " "},

		// Numeric operators
		{s: `+`, tok: influxql.ADD},
		{s: `-`, tok: influxql.SUB},
		{s: `*`, tok: influxql.MUL},
		{s: `/`, tok: influxql.DIV},

		// Logical operators
		{s: `AND`, tok: influxql.AND},
		{s: `and`, tok: influxql.AND},
		{s: `OR`, tok: influxql.OR},
		{s: `or`, tok: influxql.OR},

		{s: `=`, tok: influxql.EQ},
		{s: `<>`, tok: influxql.NEQ},
		{s: `! `, tok: influxql.ILLEGAL, lit: "!"},
		{s: `<`, tok: influxql.LT},
		{s: `<=`, tok: influxql.LTE},
		{s: `>`, tok: influxql.GT},
		{s: `>=`, tok: influxql.GTE},

		// Misc tokens
		{s: `(`, tok: influxql.LPAREN},
		{s: `)`, tok: influxql.RPAREN},
		{s: `,`, tok: influxql.COMMA},
		{s: `;`, tok: influxql.SEMICOLON},
		{s: `.`, tok: influxql.DOT},
		{s: `=~`, tok: influxql.EQREGEX},
		{s: `!~`, tok: influxql.NEQREGEX},
		{s: `:`, tok: influxql.COLON},
		{s: `::`, tok: influxql.DOUBLECOLON},

		// Identifiers
		{s: `foo`, tok: influxql.IDENT, lit: `foo`},
		{s: `_foo`, tok: influxql.IDENT, lit: `_foo`},
		{s: `Zx12_3U_-`, tok: influxql.IDENT, lit: `Zx12_3U_`},
		{s: `"foo"`, tok: influxql.IDENT, lit: `foo`},
		{s: `"foo\\bar"`, tok: influxql.IDENT, lit: `foo\bar`},
		{s: `"foo\bar"`, tok: influxql.BADESCAPE, lit: `\b`, pos: influxql.Pos{Line: 0, Char: 5}},
		{s: `"foo\"bar\""`, tok: influxql.IDENT, lit: `foo"bar"`},
		{s: `test"`, tok: influxql.BADSTRING, lit: "", pos: influxql.Pos{Line: 0, Char: 3}},
		{s: `"test`, tok: influxql.BADSTRING, lit: `test`},
		{s: `$host`, tok: influxql.BOUNDPARAM, lit: `$host`},
		{s: `$"host param"`, tok: influxql.BOUNDPARAM, lit: `$host param`},

		{s: `true`, tok: influxql.TRUE},
		{s: `false`, tok: influxql.FALSE},

		// Strings
		{s: `'testing 123!'`, tok: influxql.STRING, lit: `testing 123!`},
		{s: `'foo\nbar'`, tok: influxql.STRING, lit: "foo\nbar"},
		{s: `'foo\\bar'`, tok: influxql.STRING, lit: "foo\\bar"},
		{s: `'test`, tok: influxql.BADSTRING, lit: `test`},
		{s: "'test\nfoo", tok: influxql.BADSTRING, lit: `test`},
		{s: `'test\g'`, tok: influxql.BADESCAPE, lit: `\g`, pos: influxql.Pos{Line: 0, Char: 6}},

		// Numbers
		{s: `100`, tok: influxql.INTEGER, lit: `100`},
		{s: `-100`, tok: influxql.INTEGER, lit: `-100`},
		{s: `100.23`, tok: influxql.NUMBER, lit: `100.23`},
		{s: `+100.23`, tok: influxql.NUMBER, lit: `+100.23`},
		{s: `-100.23`, tok: influxql.NUMBER, lit: `-100.23`},
		{s: `-100.`, tok: influxql.NUMBER, lit: `-100`},
		{s: `.23`, tok: influxql.NUMBER, lit: `.23`},
		{s: `+.23`, tok: influxql.NUMBER, lit: `+.23`},
		{s: `-.23`, tok: influxql.NUMBER, lit: `-.23`},
		//{s: `.`, tok: influxql.ILLEGAL, lit: `.`},
		{s: `-.`, tok: influxql.SUB, lit: ``},
		{s: `+.`, tok: influxql.ADD, lit: ``},
		{s: `10.3s`, tok: influxql.NUMBER, lit: `10.3`},

		// Durations
		{s: `10u`, tok: influxql.DURATIONVAL, lit: `10u`},
		{s: `10µ`, tok: influxql.DURATIONVAL, lit: `10µ`},
		{s: `10ms`, tok: influxql.DURATIONVAL, lit: `10ms`},
		{s: `-1s`, tok: influxql.DURATIONVAL, lit: `-1s`},
		{s: `10m`, tok: influxql.DURATIONVAL, lit: `10m`},
		{s: `10h`, tok: influxql.DURATIONVAL, lit: `10h`},
		{s: `10d`, tok: influxql.DURATIONVAL, lit: `10d`},
		{s: `10w`, tok: influxql.DURATIONVAL, lit: `10w`},
		{s: `10x`, tok: influxql.DURATIONVAL, lit: `10x`}, // non-duration unit, but scanned as a duration value

		// Keywords
		{s: `ALL`, tok: influxql.ALL},
		{s: `ALTER`, tok: influxql.ALTER},
		{s: `AS`, tok: influxql.AS},
		{s: `ASC`, tok: influxql.ASC},
		{s: `BEGIN`, tok: influxql.BEGIN},
		{s: `BY`, tok: influxql.BY},
		{s: `CREATE`, tok: influxql.CREATE},
		{s: `CONTINUOUS`, tok: influxql.CONTINUOUS},
		{s: `DATABASE`, tok: influxql.DATABASE},
		{s: `DATABASES`, tok: influxql.DATABASES},
		{s: `DEFAULT`, tok: influxql.DEFAULT},
		{s: `DELETE`, tok: influxql.DELETE},
		{s: `DESC`, tok: influxql.DESC},
		{s: `DROP`, tok: influxql.DROP},
		{s: `DURATION`, tok: influxql.DURATION},
		{s: `END`, tok: influxql.END},
		{s: `EVERY`, tok: influxql.EVERY},
		{s: `EXPLAIN`, tok: influxql.EXPLAIN},
		{s: `FIELD`, tok: influxql.FIELD},
		{s: `FROM`, tok: influxql.FROM},
		{s: `GRANT`, tok: influxql.GRANT},
		{s: `GROUP`, tok: influxql.GROUP},
		{s: `GROUPS`, tok: influxql.GROUPS},
		{s: `INSERT`, tok: influxql.INSERT},
		{s: `INTO`, tok: influxql.INTO},
		{s: `KEY`, tok: influxql.KEY},
		{s: `KEYS`, tok: influxql.KEYS},
		{s: `KILL`, tok: influxql.KILL},
		{s: `LIMIT`, tok: influxql.LIMIT},
		{s: `SHOW`, tok: influxql.SHOW},
		{s: `SHARD`, tok: influxql.SHARD},
		{s: `SHARDS`, tok: influxql.SHARDS},
		{s: `MEASUREMENT`, tok: influxql.MEASUREMENT},
		{s: `MEASUREMENTS`, tok: influxql.MEASUREMENTS},
		{s: `OFFSET`, tok: influxql.OFFSET},
		{s: `ON`, tok: influxql.ON},
		{s: `ORDER`, tok: influxql.ORDER},
		{s: `PASSWORD`, tok: influxql.PASSWORD},
		{s: `POLICY`, tok: influxql.POLICY},
		{s: `POLICIES`, tok: influxql.POLICIES},
		{s: `PRIVILEGES`, tok: influxql.PRIVILEGES},
		{s: `QUERIES`, tok: influxql.QUERIES},
		{s: `QUERY`, tok: influxql.QUERY},
		{s: `READ`, tok: influxql.READ},
		{s: `REPLICATION`, tok: influxql.REPLICATION},
		{s: `RESAMPLE`, tok: influxql.RESAMPLE},
		{s: `RETENTION`, tok: influxql.RETENTION},
		{s: `REVOKE`, tok: influxql.REVOKE},
		{s: `SELECT`, tok: influxql.SELECT},
		{s: `SERIES`, tok: influxql.SERIES},
		{s: `TAG`, tok: influxql.TAG},
		{s: `TO`, tok: influxql.TO},
		{s: `USER`, tok: influxql.USER},
		{s: `USERS`, tok: influxql.USERS},
		{s: `VALUES`, tok: influxql.VALUES},
		{s: `WHERE`, tok: influxql.WHERE},
		{s: `WITH`, tok: influxql.WITH},
		{s: `WRITE`, tok: influxql.WRITE},
		{s: `explain`, tok: influxql.EXPLAIN}, // case insensitive
		{s: `seLECT`, tok: influxql.SELECT},   // case insensitive
	}

	for i, tt := range tests {
		s := influxql.NewScanner(strings.NewReader(tt.s))
		tok, pos, lit := s.Scan()
		if tt.tok != tok {
			t.Errorf("%d. %q token mismatch: exp=%q got=%q <%q>", i, tt.s, tt.tok, tok, lit)
		} else if tt.pos.Line != pos.Line || tt.pos.Char != pos.Char {
			t.Errorf("%d. %q pos mismatch: exp=%#v got=%#v", i, tt.s, tt.pos, pos)
		} else if tt.lit != lit {
			t.Errorf("%d. %q literal mismatch: exp=%q got=%q", i, tt.s, tt.lit, lit)
		}
	}
}

// Ensure the scanner can scan a series of tokens correctly.
func TestScanner_Scan_Multi(t *testing.T) {
	type result struct {
		tok influxql.Token
		pos influxql.Pos
		lit string
	}
	exp := []result{
		{tok: influxql.SELECT, pos: influxql.Pos{Line: 0, Char: 0}, lit: ""},
		{tok: influxql.WS, pos: influxql.Pos{Line: 0, Char: 6}, lit: " "},
		{tok: influxql.IDENT, pos: influxql.Pos{Line: 0, Char: 7}, lit: "value"},
		{tok: influxql.WS, pos: influxql.Pos{Line: 0, Char: 12}, lit: " "},
		{tok: influxql.FROM, pos: influxql.Pos{Line: 0, Char: 13}, lit: ""},
		{tok: influxql.WS, pos: influxql.Pos{Line: 0, Char: 17}, lit: " "},
		{tok: influxql.IDENT, pos: influxql.Pos{Line: 0, Char: 18}, lit: "myseries"},
		{tok: influxql.WS, pos: influxql.Pos{Line: 0, Char: 26}, lit: " "},
		{tok: influxql.WHERE, pos: influxql.Pos{Line: 0, Char: 27}, lit: ""},
		{tok: influxql.WS, pos: influxql.Pos{Line: 0, Char: 32}, lit: " "},
		{tok: influxql.IDENT, pos: influxql.Pos{Line: 0, Char: 33}, lit: "a"},
		{tok: influxql.WS, pos: influxql.Pos{Line: 0, Char: 34}, lit: " "},
		{tok: influxql.EQ, pos: influxql.Pos{Line: 0, Char: 35}, lit: ""},
		{tok: influxql.WS, pos: influxql.Pos{Line: 0, Char: 36}, lit: " "},
		{tok: influxql.STRING, pos: influxql.Pos{Line: 0, Char: 36}, lit: "b"},
		{tok: influxql.EOF, pos: influxql.Pos{Line: 0, Char: 40}, lit: ""},
	}

	// Create a scanner.
	v := `SELECT value from myseries WHERE a = 'b'`
	s := influxql.NewScanner(strings.NewReader(v))

	// Continually scan until we reach the end.
	var act []result
	for {
		tok, pos, lit := s.Scan()
		act = append(act, result{tok, pos, lit})
		if tok == influxql.EOF {
			break
		}
	}

	// Verify the token counts match.
	if len(exp) != len(act) {
		t.Fatalf("token count mismatch: exp=%d, got=%d", len(exp), len(act))
	}

	// Verify each token matches.
	for i := range exp {
		if !reflect.DeepEqual(exp[i], act[i]) {
			t.Fatalf("%d. token mismatch:\n\nexp=%#v\n\ngot=%#v", i, exp[i], act[i])
		}
	}
}

// Ensure the library can correctly scan strings.
func TestScanString(t *testing.T) {
	var tests = []struct {
		in  string
		out string
		err string
	}{
		{in: `""`, out: ``},
		{in: `"foo bar"`, out: `foo bar`},
		{in: `'foo bar'`, out: `foo bar`},
		{in: `"foo\nbar"`, out: "foo\nbar"},
		{in: `"foo\\bar"`, out: `foo\bar`},
		{in: `"foo\"bar"`, out: `foo"bar`},
		{in: `'foo\'bar'`, out: `foo'bar`},

		{in: `"foo` + "\n", out: `foo`, err: "bad string"}, // newline in string
		{in: `"foo`, out: `foo`, err: "bad string"},        // unclosed quotes
		{in: `"foo\xbar"`, out: `\x`, err: "bad escape"},   // invalid escape
	}

	for i, tt := range tests {
		out, err := influxql.ScanString(strings.NewReader(tt.in))
		if tt.err != errstring(err) {
			t.Errorf("%d. %s: error: exp=%s, got=%s", i, tt.in, tt.err, err)
		} else if tt.out != out {
			t.Errorf("%d. %s: out: exp=%s, got=%s", i, tt.in, tt.out, out)
		}
	}
}

// Test scanning regex
func TestScanRegex(t *testing.T) {
	var tests = []struct {
		in  string
		tok influxql.Token
		lit string
		err string
	}{
		{in: `/^payments\./`, tok: influxql.REGEX, lit: `^payments\.`},
		{in: `/foo\/bar/`, tok: influxql.REGEX, lit: `foo/bar`},
		{in: `/foo\\/bar/`, tok: influxql.REGEX, lit: `foo\/bar`},
		{in: `/foo\\bar/`, tok: influxql.REGEX, lit: `foo\\bar`},
		{in: `/http\:\/\/www\.example\.com/`, tok: influxql.REGEX, lit: `http\://www\.example\.com`},
	}

	for i, tt := range tests {
		s := influxql.NewScanner(strings.NewReader(tt.in))
		tok, _, lit := s.ScanRegex()
		if tok != tt.tok {
			t.Errorf("%d. %s: error:\n\texp=%s\n\tgot=%s\n", i, tt.in, tt.tok.String(), tok.String())
		}
		if lit != tt.lit {
			t.Errorf("%d. %s: error:\n\texp=%s\n\tgot=%s\n", i, tt.in, tt.lit, lit)
		}
	}
}
