package rule

import (
	"fmt"
	"go/ast"

	"strings"
	"sync"

	"github.com/mgechev/revive/lint"
)

type referenceMethod struct {
	fileName string
	id       *ast.Ident
}

type pkgMethods struct {
	pkg     *lint.Package
	methods map[string]map[string]*referenceMethod
	mu      *sync.Mutex
}

type packages struct {
	pkgs []pkgMethods
	mu   sync.Mutex
}

func (ps *packages) methodNames(lp *lint.Package) pkgMethods {
	ps.mu.Lock()

	for _, pkg := range ps.pkgs {
		if pkg.pkg == lp {
			ps.mu.Unlock()
			return pkg
		}
	}

	pkgm := pkgMethods{pkg: lp, methods: make(map[string]map[string]*referenceMethod), mu: &sync.Mutex{}}
	ps.pkgs = append(ps.pkgs, pkgm)

	ps.mu.Unlock()
	return pkgm
}

var allPkgs = packages{pkgs: make([]pkgMethods, 1)}

// ConfusingNamingRule lints method names that differ only by capitalization
type ConfusingNamingRule struct{}

// Apply applies the rule to given file.
func (r *ConfusingNamingRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure
	fileAst := file.AST
	pkgm := allPkgs.methodNames(file.Pkg)
	walker := lintConfusingNames{
		fileName: file.Name,
		pkgm:     pkgm,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(&walker, fileAst)

	return failures
}

// Name returns the rule name.
func (r *ConfusingNamingRule) Name() string {
	return "confusing-naming"
}

//checkMethodName checks if a given method/function name is similar (just case differences) to other method/function of the same struct/file.
func checkMethodName(holder string, id *ast.Ident, w *lintConfusingNames) {
	if id.Name == "init" && holder == defaultStructName {
		// ignore init functions
		return
	}

	pkgm := w.pkgm
	name := strings.ToUpper(id.Name)

	pkgm.mu.Lock()
	defer pkgm.mu.Unlock()

	if pkgm.methods[holder] != nil {
		if pkgm.methods[holder][name] != nil {
			refMethod := pkgm.methods[holder][name]
			// confusing names
			var kind string
			if holder == defaultStructName {
				kind = "function"
			} else {
				kind = "method"
			}
			var fileName string
			if w.fileName == refMethod.fileName {
				fileName = "the same source file"
			} else {
				fileName = refMethod.fileName
			}
			w.onFailure(lint.Failure{
				Failure:    fmt.Sprintf("Method '%s' differs only by capitalization to %s '%s' in %s", id.Name, kind, refMethod.id.Name, fileName),
				Confidence: 1,
				Node:       id,
				Category:   "naming",
			})

			return
		}
	} else {
		pkgm.methods[holder] = make(map[string]*referenceMethod, 1)
	}

	// update the black list
	if pkgm.methods[holder] == nil {
		println("no entry for '", holder, "'")
	}
	pkgm.methods[holder][name] = &referenceMethod{fileName: w.fileName, id: id}
}

type lintConfusingNames struct {
	fileName  string
	pkgm      pkgMethods
	onFailure func(lint.Failure)
}

const defaultStructName = "_" // used to map functions

//getStructName of a function receiver. Defaults to defaultStructName
func getStructName(r *ast.FieldList) string {
	result := defaultStructName

	if r == nil || len(r.List) < 1 {
		return result
	}

	t := r.List[0].Type

	if p, _ := t.(*ast.StarExpr); p != nil { // if a pointer receiver => dereference pointer receiver types
		t = p.X
	}

	if p, _ := t.(*ast.Ident); p != nil {
		result = p.Name
	}

	return result
}

func checkStructFields(fields *ast.FieldList, structName string, w *lintConfusingNames) {
	bl := make(map[string]bool, len(fields.List))
	for _, f := range fields.List {
		for _, id := range f.Names {
			normName := strings.ToUpper(id.Name)
			if bl[normName] {
				w.onFailure(lint.Failure{
					Failure:    fmt.Sprintf("Field '%s' differs only by capitalization to other field in the struct type %s", id.Name, structName),
					Confidence: 1,
					Node:       id,
					Category:   "naming",
				})
			} else {
				bl[normName] = true
			}
		}
	}
}

func (w *lintConfusingNames) Visit(n ast.Node) ast.Visitor {
	switch v := n.(type) {
	case *ast.FuncDecl:
		// Exclude naming warnings for functions that are exported to C but
		// not exported in the Go API.
		// See https://github.com/golang/lint/issues/144.
		if ast.IsExported(v.Name.Name) || !isCgoExported(v) {
			checkMethodName(getStructName(v.Recv), v.Name, w)
		}
	case *ast.TypeSpec:
		if s, ok := v.Type.(*ast.StructType); ok {
			checkStructFields(s.Fields, v.Name.Name, w)
		}

	default:
		// will add other checks like field names, struct names, etc.
	}

	return w
}
