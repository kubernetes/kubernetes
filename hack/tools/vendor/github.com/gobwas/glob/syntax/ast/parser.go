package ast

import (
	"errors"
	"fmt"
	"github.com/gobwas/glob/syntax/lexer"
	"unicode/utf8"
)

type Lexer interface {
	Next() lexer.Token
}

type parseFn func(*Node, Lexer) (parseFn, *Node, error)

func Parse(lexer Lexer) (*Node, error) {
	var parser parseFn

	root := NewNode(KindPattern, nil)

	var (
		tree *Node
		err  error
	)
	for parser, tree = parserMain, root; parser != nil; {
		parser, tree, err = parser(tree, lexer)
		if err != nil {
			return nil, err
		}
	}

	return root, nil
}

func parserMain(tree *Node, lex Lexer) (parseFn, *Node, error) {
	for {
		token := lex.Next()
		switch token.Type {
		case lexer.EOF:
			return nil, tree, nil

		case lexer.Error:
			return nil, tree, errors.New(token.Raw)

		case lexer.Text:
			Insert(tree, NewNode(KindText, Text{token.Raw}))
			return parserMain, tree, nil

		case lexer.Any:
			Insert(tree, NewNode(KindAny, nil))
			return parserMain, tree, nil

		case lexer.Super:
			Insert(tree, NewNode(KindSuper, nil))
			return parserMain, tree, nil

		case lexer.Single:
			Insert(tree, NewNode(KindSingle, nil))
			return parserMain, tree, nil

		case lexer.RangeOpen:
			return parserRange, tree, nil

		case lexer.TermsOpen:
			a := NewNode(KindAnyOf, nil)
			Insert(tree, a)

			p := NewNode(KindPattern, nil)
			Insert(a, p)

			return parserMain, p, nil

		case lexer.Separator:
			p := NewNode(KindPattern, nil)
			Insert(tree.Parent, p)

			return parserMain, p, nil

		case lexer.TermsClose:
			return parserMain, tree.Parent.Parent, nil

		default:
			return nil, tree, fmt.Errorf("unexpected token: %s", token)
		}
	}
	return nil, tree, fmt.Errorf("unknown error")
}

func parserRange(tree *Node, lex Lexer) (parseFn, *Node, error) {
	var (
		not   bool
		lo    rune
		hi    rune
		chars string
	)
	for {
		token := lex.Next()
		switch token.Type {
		case lexer.EOF:
			return nil, tree, errors.New("unexpected end")

		case lexer.Error:
			return nil, tree, errors.New(token.Raw)

		case lexer.Not:
			not = true

		case lexer.RangeLo:
			r, w := utf8.DecodeRuneInString(token.Raw)
			if len(token.Raw) > w {
				return nil, tree, fmt.Errorf("unexpected length of lo character")
			}
			lo = r

		case lexer.RangeBetween:
			//

		case lexer.RangeHi:
			r, w := utf8.DecodeRuneInString(token.Raw)
			if len(token.Raw) > w {
				return nil, tree, fmt.Errorf("unexpected length of lo character")
			}

			hi = r

			if hi < lo {
				return nil, tree, fmt.Errorf("hi character '%s' should be greater than lo '%s'", string(hi), string(lo))
			}

		case lexer.Text:
			chars = token.Raw

		case lexer.RangeClose:
			isRange := lo != 0 && hi != 0
			isChars := chars != ""

			if isChars == isRange {
				return nil, tree, fmt.Errorf("could not parse range")
			}

			if isRange {
				Insert(tree, NewNode(KindRange, Range{
					Lo:  lo,
					Hi:  hi,
					Not: not,
				}))
			} else {
				Insert(tree, NewNode(KindList, List{
					Chars: chars,
					Not:   not,
				}))
			}

			return parserMain, tree, nil
		}
	}
}
