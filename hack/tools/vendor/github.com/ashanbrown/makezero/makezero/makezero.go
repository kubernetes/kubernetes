// Package makezero provides a linter for appends to slices initialized with non-zero length.
package makezero

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"go/types"
	"log"
	"regexp"
)

// a decl might include multiple var,
// so var name with decl make final uniq obj.
type uniqDecl struct {
	varName string
	decl    interface{}
}

type Issue interface {
	Details() string
	Pos() token.Pos
	Position() token.Position
	String() string
}

type AppendIssue struct {
	name     string
	pos      token.Pos
	position token.Position
}

func (a AppendIssue) Details() string {
	return fmt.Sprintf("append to slice `%s` with non-zero initialized length", a.name)
}

func (a AppendIssue) Pos() token.Pos {
	return a.pos
}

func (a AppendIssue) Position() token.Position {
	return a.position
}

func (a AppendIssue) String() string { return toString(a) }

type MustHaveNonZeroInitLenIssue struct {
	name     string
	pos      token.Pos
	position token.Position
}

func (i MustHaveNonZeroInitLenIssue) Details() string {
	return fmt.Sprintf("slice `%s` does not have non-zero initial length", i.name)
}

func (i MustHaveNonZeroInitLenIssue) Pos() token.Pos {
	return i.pos
}

func (i MustHaveNonZeroInitLenIssue) Position() token.Position {
	return i.position
}

func (i MustHaveNonZeroInitLenIssue) String() string { return toString(i) }

func toString(i Issue) string {
	return fmt.Sprintf("%s at %s", i.Details(), i.Position())
}

type visitor struct {
	initLenMustBeZero bool

	comments []*ast.CommentGroup // comments to apply during this visit
	info     *types.Info

	nonZeroLengthSliceDecls map[uniqDecl]struct{}
	fset                    *token.FileSet
	issues                  []Issue
}

type Linter struct {
	initLenMustBeZero bool
}

func NewLinter(initialLengthMustBeZero bool) *Linter {
	return &Linter{
		initLenMustBeZero: initialLengthMustBeZero,
	}
}

func (l Linter) Run(fset *token.FileSet, info *types.Info, nodes ...ast.Node) ([]Issue, error) {
	var issues []Issue
	for _, node := range nodes {
		var comments []*ast.CommentGroup
		if file, ok := node.(*ast.File); ok {
			comments = file.Comments
		}
		visitor := visitor{
			nonZeroLengthSliceDecls: make(map[uniqDecl]struct{}),
			initLenMustBeZero:       l.initLenMustBeZero,
			info:                    info,
			fset:                    fset,
			comments:                comments,
		}
		ast.Walk(&visitor, node)
		issues = append(issues, visitor.issues...)
	}
	return issues, nil
}

func (v *visitor) Visit(node ast.Node) ast.Visitor {
	switch node := node.(type) {
	case *ast.CallExpr:
		fun, ok := node.Fun.(*ast.Ident)
		if !ok || fun.Name != "append" {
			break
		}
		if sliceIdent, ok := node.Args[0].(*ast.Ident); ok &&
			v.hasNonZeroInitialLength(sliceIdent) &&
			!v.hasNoLintOnSameLine(fun) {
			v.issues = append(v.issues,
				AppendIssue{
					name:     sliceIdent.Name,
					pos:      fun.Pos(),
					position: v.fset.Position(fun.Pos()),
				})
		}
	case *ast.AssignStmt:
		for i, right := range node.Rhs {
			if right, ok := right.(*ast.CallExpr); ok {
				fun, ok := right.Fun.(*ast.Ident)
				if !ok || fun.Name != "make" {
					continue
				}
				left := node.Lhs[i]
				if len(right.Args) == 2 {
					// ignore if not a slice or it has explicit zero length
					if !v.isSlice(right.Args[0]) {
						continue
					} else if lit, ok := right.Args[1].(*ast.BasicLit); ok && lit.Kind == token.INT && lit.Value == "0" {
						continue
					}
					if v.initLenMustBeZero && !v.hasNoLintOnSameLine(fun) {
						v.issues = append(v.issues, MustHaveNonZeroInitLenIssue{
							name:     v.textFor(left),
							pos:      node.Pos(),
							position: v.fset.Position(node.Pos()),
						})
					}
					v.recordNonZeroLengthSlices(left)
				}
			}
		}
	}
	return v
}

func (v *visitor) textFor(node ast.Node) string {
	typeBuf := new(bytes.Buffer)
	if err := printer.Fprint(typeBuf, v.fset, node); err != nil {
		log.Fatalf("ERROR: unable to print type: %s", err)
	}
	return typeBuf.String()
}

func (v *visitor) hasNonZeroInitialLength(ident *ast.Ident) bool {
	if ident.Obj == nil {
		log.Printf("WARNING: could not determine with %q at %s is a slice (missing object type)",
			ident.Name, v.fset.Position(ident.Pos()).String())
		return false
	}
	_, exists := v.nonZeroLengthSliceDecls[uniqDecl{
		varName: ident.Obj.Name,
		decl:    ident.Obj.Decl,
	}]
	return exists
}

func (v *visitor) recordNonZeroLengthSlices(node ast.Node) {
	ident, ok := node.(*ast.Ident)
	if !ok {
		return
	}
	if ident.Obj == nil {
		return
	}
	v.nonZeroLengthSliceDecls[uniqDecl{
		varName: ident.Obj.Name,
		decl:    ident.Obj.Decl,
	}] = struct{}{}
}

func (v *visitor) isSlice(node ast.Node) bool {
	// determine type if this is a user-defined type
	if ident, ok := node.(*ast.Ident); ok {
		obj := ident.Obj
		if obj == nil {
			if v.info != nil {
				_, ok := v.info.ObjectOf(ident).Type().(*types.Slice)
				return ok
			}
			return false
		}
		spec, ok := obj.Decl.(*ast.TypeSpec)
		if !ok {
			return false
		}
		node = spec.Type
	}

	if node, ok := node.(*ast.ArrayType); ok {
		return node.Len == nil // only slices have zero length
	}
	return false
}

func (v *visitor) hasNoLintOnSameLine(node ast.Node) bool {
	nolint := regexp.MustCompile(`^\s*nozero\b`)
	nodePos := v.fset.Position(node.Pos())
	for _, c := range v.comments {
		commentPos := v.fset.Position(c.Pos())
		if commentPos.Line == nodePos.Line && nolint.MatchString(c.Text()) {
			return true
		}
	}
	return false
}
