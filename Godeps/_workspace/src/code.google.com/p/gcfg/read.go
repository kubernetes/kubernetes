package gcfg

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"
)

import (
	"code.google.com/p/gcfg/scanner"
	"code.google.com/p/gcfg/token"
)

var unescape = map[rune]rune{'\\': '\\', '"': '"', 'n': '\n', 't': '\t'}

// no error: invalid literals should be caught by scanner
func unquote(s string) string {
	u, q, esc := make([]rune, 0, len(s)), false, false
	for _, c := range s {
		if esc {
			uc, ok := unescape[c]
			switch {
			case ok:
				u = append(u, uc)
				fallthrough
			case !q && c == '\n':
				esc = false
				continue
			}
			panic("invalid escape sequence")
		}
		switch c {
		case '"':
			q = !q
		case '\\':
			esc = true
		default:
			u = append(u, c)
		}
	}
	if q {
		panic("missing end quote")
	}
	if esc {
		panic("invalid escape sequence")
	}
	return string(u)
}

func readInto(config interface{}, fset *token.FileSet, file *token.File, src []byte) error {
	var s scanner.Scanner
	var errs scanner.ErrorList
	s.Init(file, src, func(p token.Position, m string) { errs.Add(p, m) }, 0)
	sect, sectsub := "", ""
	pos, tok, lit := s.Scan()
	errfn := func(msg string) error {
		return fmt.Errorf("%s: %s", fset.Position(pos), msg)
	}
	for {
		if errs.Len() > 0 {
			return errs.Err()
		}
		switch tok {
		case token.EOF:
			return nil
		case token.EOL, token.COMMENT:
			pos, tok, lit = s.Scan()
		case token.LBRACK:
			pos, tok, lit = s.Scan()
			if errs.Len() > 0 {
				return errs.Err()
			}
			if tok != token.IDENT {
				return errfn("expected section name")
			}
			sect, sectsub = lit, ""
			pos, tok, lit = s.Scan()
			if errs.Len() > 0 {
				return errs.Err()
			}
			if tok == token.STRING {
				sectsub = unquote(lit)
				if sectsub == "" {
					return errfn("empty subsection name")
				}
				pos, tok, lit = s.Scan()
				if errs.Len() > 0 {
					return errs.Err()
				}
			}
			if tok != token.RBRACK {
				if sectsub == "" {
					return errfn("expected subsection name or right bracket")
				}
				return errfn("expected right bracket")
			}
			pos, tok, lit = s.Scan()
			if tok != token.EOL && tok != token.EOF && tok != token.COMMENT {
				return errfn("expected EOL, EOF, or comment")
			}
		case token.IDENT:
			if sect == "" {
				return errfn("expected section header")
			}
			n := lit
			pos, tok, lit = s.Scan()
			if errs.Len() > 0 {
				return errs.Err()
			}
			blank, v := tok == token.EOF || tok == token.EOL || tok == token.COMMENT, ""
			if !blank {
				if tok != token.ASSIGN {
					return errfn("expected '='")
				}
				pos, tok, lit = s.Scan()
				if errs.Len() > 0 {
					return errs.Err()
				}
				if tok != token.STRING {
					return errfn("expected value")
				}
				v = unquote(lit)
				pos, tok, lit = s.Scan()
				if errs.Len() > 0 {
					return errs.Err()
				}
				if tok != token.EOL && tok != token.EOF && tok != token.COMMENT {
					return errfn("expected EOL, EOF, or comment")
				}
			}
			err := set(config, sect, sectsub, n, blank, v)
			if err != nil {
				return err
			}
		default:
			if sect == "" {
				return errfn("expected section header")
			}
			return errfn("expected section header or variable declaration")
		}
	}
	panic("never reached")
}

// ReadInto reads gcfg formatted data from reader and sets the values into the
// corresponding fields in config.
func ReadInto(config interface{}, reader io.Reader) error {
	src, err := ioutil.ReadAll(reader)
	if err != nil {
		return err
	}
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(src))
	return readInto(config, fset, file, src)
}

// ReadStringInto reads gcfg formatted data from str and sets the values into
// the corresponding fields in config.
func ReadStringInto(config interface{}, str string) error {
	r := strings.NewReader(str)
	return ReadInto(config, r)
}

// ReadFileInto reads gcfg formatted data from the file filename and sets the
// values into the corresponding fields in config.
func ReadFileInto(config interface{}, filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	src, err := ioutil.ReadAll(f)
	if err != nil {
		return err
	}
	fset := token.NewFileSet()
	file := fset.AddFile(filename, fset.Base(), len(src))
	return readInto(config, fset, file, src)
}
