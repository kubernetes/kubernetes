package influxql

import (
	"strings"
)

// Token is a lexical token of the InfluxQL language.
type Token int

const (
	// Special tokens
	ILLEGAL Token = iota
	EOF
	WS

	literal_beg
	// Literals
	IDENT        // main
	NUMBER       // 12345.67
	DURATION_VAL // 13h
	STRING       // "abc"
	BADSTRING    // "abc
	BADESCAPE    // \q
	TRUE         // true
	FALSE        // false
	REGEX        // Regular expressions
	BADREGEX     // `.*
	literal_end

	operator_beg
	// Operators
	ADD // +
	SUB // -
	MUL // *
	DIV // /

	AND // AND
	OR  // OR

	EQ       // =
	NEQ      // !=
	EQREGEX  // =~
	NEQREGEX // !~
	LT       // <
	LTE      // <=
	GT       // >
	GTE      // >=
	operator_end

	LPAREN    // (
	RPAREN    // )
	COMMA     // ,
	COLON     // :
	SEMICOLON // ;
	DOT       // .

	keyword_beg
	// Keywords
	ALL
	ALTER
	ANY
	AS
	ASC
	BEGIN
	BY
	CREATE
	CONTINUOUS
	DATABASE
	DATABASES
	DEFAULT
	DELETE
	DESC
	DESTINATIONS
	DIAGNOSTICS
	DISTINCT
	DROP
	DURATION
	END
	EXISTS
	EXPLAIN
	FIELD
	FOR
	FORCE
	FROM
	GRANT
	GRANTS
	GROUP
	GROUPS
	IF
	IN
	INF
	INNER
	INSERT
	INTO
	KEY
	KEYS
	LIMIT
	MEASUREMENT
	MEASUREMENTS
	NAME
	NOT
	OFFSET
	ON
	ORDER
	PASSWORD
	POLICY
	POLICIES
	PRIVILEGES
	QUERIES
	QUERY
	READ
	REPLICATION
	RETENTION
	REVOKE
	SELECT
	SERIES
	SERVER
	SERVERS
	SET
	SHOW
	SHARD
	SHARDS
	SLIMIT
	SOFFSET
	STATS
	SUBSCRIPTION
	SUBSCRIPTIONS
	TAG
	TO
	USER
	USERS
	VALUES
	WHERE
	WITH
	WRITE
	keyword_end
)

var tokens = [...]string{
	ILLEGAL: "ILLEGAL",
	EOF:     "EOF",
	WS:      "WS",

	IDENT:        "IDENT",
	NUMBER:       "NUMBER",
	DURATION_VAL: "DURATION_VAL",
	STRING:       "STRING",
	BADSTRING:    "BADSTRING",
	BADESCAPE:    "BADESCAPE",
	TRUE:         "TRUE",
	FALSE:        "FALSE",
	REGEX:        "REGEX",

	ADD: "+",
	SUB: "-",
	MUL: "*",
	DIV: "/",

	AND: "AND",
	OR:  "OR",

	EQ:       "=",
	NEQ:      "!=",
	EQREGEX:  "=~",
	NEQREGEX: "!~",
	LT:       "<",
	LTE:      "<=",
	GT:       ">",
	GTE:      ">=",

	LPAREN:    "(",
	RPAREN:    ")",
	COMMA:     ",",
	COLON:     ":",
	SEMICOLON: ";",
	DOT:       ".",

	ALL:           "ALL",
	ALTER:         "ALTER",
	ANY:           "ANY",
	AS:            "AS",
	ASC:           "ASC",
	BEGIN:         "BEGIN",
	BY:            "BY",
	CREATE:        "CREATE",
	CONTINUOUS:    "CONTINUOUS",
	DATABASE:      "DATABASE",
	DATABASES:     "DATABASES",
	DEFAULT:       "DEFAULT",
	DELETE:        "DELETE",
	DESC:          "DESC",
	DESTINATIONS:  "DESTINATIONS",
	DIAGNOSTICS:   "DIAGNOSTICS",
	DISTINCT:      "DISTINCT",
	DROP:          "DROP",
	DURATION:      "DURATION",
	END:           "END",
	EXISTS:        "EXISTS",
	EXPLAIN:       "EXPLAIN",
	FIELD:         "FIELD",
	FOR:           "FOR",
	FORCE:         "FORCE",
	FROM:          "FROM",
	GRANT:         "GRANT",
	GRANTS:        "GRANTS",
	GROUP:         "GROUP",
	GROUPS:        "GROUPS",
	IF:            "IF",
	IN:            "IN",
	INF:           "INF",
	INNER:         "INNER",
	INSERT:        "INSERT",
	INTO:          "INTO",
	KEY:           "KEY",
	KEYS:          "KEYS",
	LIMIT:         "LIMIT",
	MEASUREMENT:   "MEASUREMENT",
	MEASUREMENTS:  "MEASUREMENTS",
	NAME:          "NAME",
	NOT:           "NOT",
	OFFSET:        "OFFSET",
	ON:            "ON",
	ORDER:         "ORDER",
	PASSWORD:      "PASSWORD",
	POLICY:        "POLICY",
	POLICIES:      "POLICIES",
	PRIVILEGES:    "PRIVILEGES",
	QUERIES:       "QUERIES",
	QUERY:         "QUERY",
	READ:          "READ",
	REPLICATION:   "REPLICATION",
	RETENTION:     "RETENTION",
	REVOKE:        "REVOKE",
	SELECT:        "SELECT",
	SERIES:        "SERIES",
	SERVER:        "SERVER",
	SERVERS:       "SERVERS",
	SET:           "SET",
	SHOW:          "SHOW",
	SHARD:         "SHARD",
	SHARDS:        "SHARDS",
	SLIMIT:        "SLIMIT",
	SOFFSET:       "SOFFSET",
	STATS:         "STATS",
	SUBSCRIPTION:  "SUBSCRIPTION",
	SUBSCRIPTIONS: "SUBSCRIPTIONS",
	TAG:           "TAG",
	TO:            "TO",
	USER:          "USER",
	USERS:         "USERS",
	VALUES:        "VALUES",
	WHERE:         "WHERE",
	WITH:          "WITH",
	WRITE:         "WRITE",
}

var keywords map[string]Token

func init() {
	keywords = make(map[string]Token)
	for tok := keyword_beg + 1; tok < keyword_end; tok++ {
		keywords[strings.ToLower(tokens[tok])] = tok
	}
	for _, tok := range []Token{AND, OR} {
		keywords[strings.ToLower(tokens[tok])] = tok
	}
	keywords["true"] = TRUE
	keywords["false"] = FALSE
}

// String returns the string representation of the token.
func (tok Token) String() string {
	if tok >= 0 && tok < Token(len(tokens)) {
		return tokens[tok]
	}
	return ""
}

// Precedence returns the operator precedence of the binary operator token.
func (tok Token) Precedence() int {
	switch tok {
	case OR:
		return 1
	case AND:
		return 2
	case EQ, NEQ, EQREGEX, NEQREGEX, LT, LTE, GT, GTE:
		return 3
	case ADD, SUB:
		return 4
	case MUL, DIV:
		return 5
	}
	return 0
}

// isOperator returns true for operator tokens.
func (tok Token) isOperator() bool { return tok > operator_beg && tok < operator_end }

// tokstr returns a literal if provided, otherwise returns the token string.
func tokstr(tok Token, lit string) string {
	if lit != "" {
		return lit
	}
	return tok.String()
}

// Lookup returns the token associated with a given string.
func Lookup(ident string) Token {
	if tok, ok := keywords[strings.ToLower(ident)]; ok {
		return tok
	}
	return IDENT
}

// Pos specifies the line and character position of a token.
// The Char and Line are both zero-based indexes.
type Pos struct {
	Line int
	Char int
}
