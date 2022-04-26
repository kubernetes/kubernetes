package glob

import (
	"github.com/gobwas/glob/compiler"
	"github.com/gobwas/glob/syntax"
)

// Glob represents compiled glob pattern.
type Glob interface {
	Match(string) bool
}

// Compile creates Glob for given pattern and strings (if any present after pattern) as separators.
// The pattern syntax is:
//
//    pattern:
//        { term }
//
//    term:
//        `*`         matches any sequence of non-separator characters
//        `**`        matches any sequence of characters
//        `?`         matches any single non-separator character
//        `[` [ `!` ] { character-range } `]`
//                    character class (must be non-empty)
//        `{` pattern-list `}`
//                    pattern alternatives
//        c           matches character c (c != `*`, `**`, `?`, `\`, `[`, `{`, `}`)
//        `\` c       matches character c
//
//    character-range:
//        c           matches character c (c != `\\`, `-`, `]`)
//        `\` c       matches character c
//        lo `-` hi   matches character c for lo <= c <= hi
//
//    pattern-list:
//        pattern { `,` pattern }
//                    comma-separated (without spaces) patterns
//
func Compile(pattern string, separators ...rune) (Glob, error) {
	ast, err := syntax.Parse(pattern)
	if err != nil {
		return nil, err
	}

	matcher, err := compiler.Compile(ast, separators)
	if err != nil {
		return nil, err
	}

	return matcher, nil
}

// MustCompile is the same as Compile, except that if Compile returns error, this will panic
func MustCompile(pattern string, separators ...rune) Glob {
	g, err := Compile(pattern, separators...)
	if err != nil {
		panic(err)
	}

	return g
}

// QuoteMeta returns a string that quotes all glob pattern meta characters
// inside the argument text; For example, QuoteMeta(`{foo*}`) returns `\[foo\*\]`.
func QuoteMeta(s string) string {
	b := make([]byte, 2*len(s))

	// a byte loop is correct because all meta characters are ASCII
	j := 0
	for i := 0; i < len(s); i++ {
		if syntax.Special(s[i]) {
			b[j] = '\\'
			j++
		}
		b[j] = s[i]
		j++
	}

	return string(b[0:j])
}
