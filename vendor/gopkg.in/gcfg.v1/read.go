package gcfg

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"

	"gopkg.in/gcfg.v1/scanner"
	"gopkg.in/gcfg.v1/token"
	"gopkg.in/warnings.v0"
)

var unescape = map[rune]rune{'\\': '\\', '"': '"', 'n': '\n', 't': '\t'}
var utf8Bom = []byte("\ufeff")

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

func readIntoPass(c *warnings.Collector, config interface{}, fset *token.FileSet,
	file *token.File, src []byte, subsectPass bool) error {
	//
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
			if err := c.Collect(errs.Err()); err != nil {
				return err
			}
		}
		switch tok {
		case token.EOF:
			return nil
		case token.EOL, token.COMMENT:
			pos, tok, lit = s.Scan()
		case token.LBRACK:
			pos, tok, lit = s.Scan()
			if errs.Len() > 0 {
				if err := c.Collect(errs.Err()); err != nil {
					return err
				}
			}
			if tok != token.IDENT {
				if err := c.Collect(errfn("expected section name")); err != nil {
					return err
				}
			}
			sect, sectsub = lit, ""
			pos, tok, lit = s.Scan()
			if errs.Len() > 0 {
				if err := c.Collect(errs.Err()); err != nil {
					return err
				}
			}
			if tok == token.STRING {
				sectsub = unquote(lit)
				if sectsub == "" {
					if err := c.Collect(errfn("empty subsection name")); err != nil {
						return err
					}
				}
				pos, tok, lit = s.Scan()
				if errs.Len() > 0 {
					if err := c.Collect(errs.Err()); err != nil {
						return err
					}
				}
			}
			if tok != token.RBRACK {
				if sectsub == "" {
					if err := c.Collect(errfn("expected subsection name or right bracket")); err != nil {
						return err
					}
				}
				if err := c.Collect(errfn("expected right bracket")); err != nil {
					return err
				}
			}
			pos, tok, lit = s.Scan()
			if tok != token.EOL && tok != token.EOF && tok != token.COMMENT {
				if err := c.Collect(errfn("expected EOL, EOF, or comment")); err != nil {
					return err
				}
			}
			// If a section/subsection header was found, ensure a
			// container object is created, even if there are no
			// variables further down.
			err := c.Collect(set(c, config, sect, sectsub, "", true, "", subsectPass))
			if err != nil {
				return err
			}
		case token.IDENT:
			if sect == "" {
				if err := c.Collect(errfn("expected section header")); err != nil {
					return err
				}
			}
			n := lit
			pos, tok, lit = s.Scan()
			if errs.Len() > 0 {
				return errs.Err()
			}
			blank, v := tok == token.EOF || tok == token.EOL || tok == token.COMMENT, ""
			if !blank {
				if tok != token.ASSIGN {
					if err := c.Collect(errfn("expected '='")); err != nil {
						return err
					}
				}
				pos, tok, lit = s.Scan()
				if errs.Len() > 0 {
					if err := c.Collect(errs.Err()); err != nil {
						return err
					}
				}
				if tok != token.STRING {
					if err := c.Collect(errfn("expected value")); err != nil {
						return err
					}
				}
				v = unquote(lit)
				pos, tok, lit = s.Scan()
				if errs.Len() > 0 {
					if err := c.Collect(errs.Err()); err != nil {
						return err
					}
				}
				if tok != token.EOL && tok != token.EOF && tok != token.COMMENT {
					if err := c.Collect(errfn("expected EOL, EOF, or comment")); err != nil {
						return err
					}
				}
			}
			err := set(c, config, sect, sectsub, n, blank, v, subsectPass)
			if err != nil {
				return err
			}
		default:
			if sect == "" {
				if err := c.Collect(errfn("expected section header")); err != nil {
					return err
				}
			}
			if err := c.Collect(errfn("expected section header or variable declaration")); err != nil {
				return err
			}
		}
	}
}

func readInto(config interface{}, fset *token.FileSet, file *token.File,
	src []byte) error {
	//
	c := warnings.NewCollector(isFatal)
	err := readIntoPass(c, config, fset, file, src, false)
	if err != nil {
		return err
	}
	err = readIntoPass(c, config, fset, file, src, true)
	if err != nil {
		return err
	}
	return c.Done()
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
//
// For compatibility with files created on Windows, the ReadFileInto skips a
// single leading UTF8 BOM sequence if it exists.
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

	// Skips a single leading UTF8 BOM sequence if it exists.
	src = skipLeadingUtf8Bom(src)

	fset := token.NewFileSet()
	file := fset.AddFile(filename, fset.Base(), len(src))
	return readInto(config, fset, file, src)
}

func skipLeadingUtf8Bom(src []byte) []byte {
	lengthUtf8Bom := len(utf8Bom)

	if len(src) >= lengthUtf8Bom {
		if bytes.Equal(src[:lengthUtf8Bom], utf8Bom) {
			return src[lengthUtf8Bom:]
		}
	}
	return src
}
