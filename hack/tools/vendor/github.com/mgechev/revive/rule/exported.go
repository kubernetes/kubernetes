package rule

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/mgechev/revive/lint"
)

// ExportedRule lints given else constructs.
type ExportedRule struct {
	configured             bool
	checkPrivateReceivers  bool
	disableStutteringCheck bool
	stuttersMsg            string
}

// Apply applies the rule to given file.
func (r *ExportedRule) Apply(file *lint.File, args lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	if isTest(file) {
		return failures
	}

	if !r.configured {
		var sayRepetitiveInsteadOfStutters bool
		r.checkPrivateReceivers, r.disableStutteringCheck, sayRepetitiveInsteadOfStutters = r.getConf(args)
		r.stuttersMsg = "stutters"
		if sayRepetitiveInsteadOfStutters {
			r.stuttersMsg = "is repetitive"
		}

		r.configured = true
	}

	fileAst := file.AST
	walker := lintExported{
		file:    file,
		fileAst: fileAst,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
		genDeclMissingComments: make(map[*ast.GenDecl]bool),
		checkPrivateReceivers:  r.checkPrivateReceivers,
		disableStutteringCheck: r.disableStutteringCheck,
		stuttersMsg:            r.stuttersMsg,
	}

	ast.Walk(&walker, fileAst)

	return failures
}

// Name returns the rule name.
func (r *ExportedRule) Name() string {
	return "exported"
}

func (r *ExportedRule) getConf(args lint.Arguments) (checkPrivateReceivers bool, disableStutteringCheck bool, sayRepetitiveInsteadOfStutters bool) {
	// if any, we expect a slice of strings as configuration
	if len(args) < 1 {
		return
	}
	for _, flag := range args {
		flagStr, ok := flag.(string)
		if !ok {
			panic(fmt.Sprintf("Invalid argument for the %s rule: expecting a string, got %T", r.Name(), flag))
		}

		switch flagStr {
		case "checkPrivateReceivers":
			checkPrivateReceivers = true
		case "disableStutteringCheck":
			disableStutteringCheck = true
		case "sayRepetitiveInsteadOfStutters":
			sayRepetitiveInsteadOfStutters = true
		default:
			panic(fmt.Sprintf("Unknown configuration flag %s for %s rule", flagStr, r.Name()))
		}
	}

	return
}

type lintExported struct {
	file                   *lint.File
	fileAst                *ast.File
	lastGen                *ast.GenDecl
	genDeclMissingComments map[*ast.GenDecl]bool
	onFailure              func(lint.Failure)
	checkPrivateReceivers  bool
	disableStutteringCheck bool
	stuttersMsg            string
}

func (w *lintExported) lintFuncDoc(fn *ast.FuncDecl) {
	if !ast.IsExported(fn.Name.Name) {
		// func is unexported
		return
	}
	kind := "function"
	name := fn.Name.Name
	if fn.Recv != nil && len(fn.Recv.List) > 0 {
		// method
		kind = "method"
		recv := receiverType(fn)
		if !w.checkPrivateReceivers && !ast.IsExported(recv) {
			// receiver is unexported
			return
		}
		if commonMethods[name] {
			return
		}
		switch name {
		case "Len", "Less", "Swap":
			if w.file.Pkg.Sortable[recv] {
				return
			}
		}
		name = recv + "." + name
	}
	if fn.Doc == nil {
		w.onFailure(lint.Failure{
			Node:       fn,
			Confidence: 1,
			Category:   "comments",
			Failure:    fmt.Sprintf("exported %s %s should have comment or be unexported", kind, name),
		})
		return
	}
	s := normalizeText(fn.Doc.Text())
	prefix := fn.Name.Name + " "
	if !strings.HasPrefix(s, prefix) {
		w.onFailure(lint.Failure{
			Node:       fn.Doc,
			Confidence: 0.8,
			Category:   "comments",
			Failure:    fmt.Sprintf(`comment on exported %s %s should be of the form "%s..."`, kind, name, prefix),
		})
	}
}

func (w *lintExported) checkStutter(id *ast.Ident, thing string) {
	if w.disableStutteringCheck {
		return
	}

	pkg, name := w.fileAst.Name.Name, id.Name
	if !ast.IsExported(name) {
		// unexported name
		return
	}
	// A name stutters if the package name is a strict prefix
	// and the next character of the name starts a new word.
	if len(name) <= len(pkg) {
		// name is too short to stutter.
		// This permits the name to be the same as the package name.
		return
	}
	if !strings.EqualFold(pkg, name[:len(pkg)]) {
		return
	}
	// We can assume the name is well-formed UTF-8.
	// If the next rune after the package name is uppercase or an underscore
	// the it's starting a new word and thus this name stutters.
	rem := name[len(pkg):]
	if next, _ := utf8.DecodeRuneInString(rem); next == '_' || unicode.IsUpper(next) {
		w.onFailure(lint.Failure{
			Node:       id,
			Confidence: 0.8,
			Category:   "naming",
			Failure:    fmt.Sprintf("%s name will be used as %s.%s by other packages, and that %s; consider calling this %s", thing, pkg, name, w.stuttersMsg, rem),
		})
	}
}

func (w *lintExported) lintTypeDoc(t *ast.TypeSpec, doc *ast.CommentGroup) {
	if !ast.IsExported(t.Name.Name) {
		return
	}
	if doc == nil {
		w.onFailure(lint.Failure{
			Node:       t,
			Confidence: 1,
			Category:   "comments",
			Failure:    fmt.Sprintf("exported type %v should have comment or be unexported", t.Name),
		})
		return
	}

	s := normalizeText(doc.Text())
	articles := [...]string{"A", "An", "The", "This"}
	for _, a := range articles {
		if t.Name.Name == a {
			continue
		}
		if strings.HasPrefix(s, a+" ") {
			s = s[len(a)+1:]
			break
		}
	}
	if !strings.HasPrefix(s, t.Name.Name+" ") {
		w.onFailure(lint.Failure{
			Node:       doc,
			Confidence: 1,
			Category:   "comments",
			Failure:    fmt.Sprintf(`comment on exported type %v should be of the form "%v ..." (with optional leading article)`, t.Name, t.Name),
		})
	}
}

func (w *lintExported) lintValueSpecDoc(vs *ast.ValueSpec, gd *ast.GenDecl, genDeclMissingComments map[*ast.GenDecl]bool) {
	kind := "var"
	if gd.Tok == token.CONST {
		kind = "const"
	}

	if len(vs.Names) > 1 {
		// Check that none are exported except for the first.
		for _, n := range vs.Names[1:] {
			if ast.IsExported(n.Name) {
				w.onFailure(lint.Failure{
					Category:   "comments",
					Confidence: 1,
					Failure:    fmt.Sprintf("exported %s %s should have its own declaration", kind, n.Name),
					Node:       vs,
				})
				return
			}
		}
	}

	// Only one name.
	name := vs.Names[0].Name
	if !ast.IsExported(name) {
		return
	}

	if vs.Doc == nil && gd.Doc == nil {
		if genDeclMissingComments[gd] {
			return
		}
		block := ""
		if kind == "const" && gd.Lparen.IsValid() {
			block = " (or a comment on this block)"
		}
		w.onFailure(lint.Failure{
			Confidence: 1,
			Node:       vs,
			Category:   "comments",
			Failure:    fmt.Sprintf("exported %s %s should have comment%s or be unexported", kind, name, block),
		})
		genDeclMissingComments[gd] = true
		return
	}
	// If this GenDecl has parens and a comment, we don't check its comment form.
	if gd.Doc != nil && gd.Lparen.IsValid() {
		return
	}
	// The relevant text to check will be on either vs.Doc or gd.Doc.
	// Use vs.Doc preferentially.
	doc := vs.Doc
	if doc == nil {
		doc = gd.Doc
	}
	prefix := name + " "
	s := normalizeText(doc.Text())
	if !strings.HasPrefix(s, prefix) {
		w.onFailure(lint.Failure{
			Confidence: 1,
			Node:       doc,
			Category:   "comments",
			Failure:    fmt.Sprintf(`comment on exported %s %s should be of the form "%s..."`, kind, name, prefix),
		})
	}
}

// normalizeText is a helper function that normalizes comment strings by:
// * removing one leading space
//
// This function is needed because ast.CommentGroup.Text() does not handle //-style and /*-style comments uniformly
func normalizeText(t string) string {
	return strings.TrimPrefix(t, " ")
}

func (w *lintExported) Visit(n ast.Node) ast.Visitor {
	switch v := n.(type) {
	case *ast.GenDecl:
		if v.Tok == token.IMPORT {
			return nil
		}
		// token.CONST, token.TYPE or token.VAR
		w.lastGen = v
		return w
	case *ast.FuncDecl:
		w.lintFuncDoc(v)
		if v.Recv == nil {
			// Only check for stutter on functions, not methods.
			// Method names are not used package-qualified.
			w.checkStutter(v.Name, "func")
		}
		// Don't proceed inside funcs.
		return nil
	case *ast.TypeSpec:
		// inside a GenDecl, which usually has the doc
		doc := v.Doc
		if doc == nil {
			doc = w.lastGen.Doc
		}
		w.lintTypeDoc(v, doc)
		w.checkStutter(v.Name, "type")
		// Don't proceed inside types.
		return nil
	case *ast.ValueSpec:
		w.lintValueSpecDoc(v, w.lastGen, w.genDeclMissingComments)
		return nil
	}
	return w
}
