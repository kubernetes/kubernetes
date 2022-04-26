package analyzer

import (
	"go/ast"
	"go/token"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

// New returns new errname analyzer.
func New() *analysis.Analyzer {
	return &analysis.Analyzer{
		Name:     "errname",
		Doc:      "Checks that sentinel errors are prefixed with the `Err` and error types are suffixed with the `Error`.",
		Run:      run,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	}
}

type stringSet = map[string]struct{}

var (
	imports = []ast.Node{(*ast.ImportSpec)(nil)}
	types   = []ast.Node{(*ast.TypeSpec)(nil)}
	funcs   = []ast.Node{(*ast.FuncDecl)(nil)}
)

func run(pass *analysis.Pass) (interface{}, error) {
	insp := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	pkgAliases := map[string]string{}
	insp.Preorder(imports, func(node ast.Node) {
		i := node.(*ast.ImportSpec)
		if n := i.Name; n != nil && i.Path != nil {
			if path, err := strconv.Unquote(i.Path.Value); err == nil {
				pkgAliases[n.Name] = getPkgFromPath(path)
			}
		}
	})

	allTypes := stringSet{}
	typesSpecs := map[string]*ast.TypeSpec{}
	insp.Preorder(types, func(node ast.Node) {
		t := node.(*ast.TypeSpec)
		allTypes[t.Name.Name] = struct{}{}
		typesSpecs[t.Name.Name] = t
	})

	errorTypes := stringSet{}
	insp.Preorder(funcs, func(node ast.Node) {
		f := node.(*ast.FuncDecl)
		t, ok := isMethodError(f)
		if !ok {
			return
		}
		errorTypes[t] = struct{}{}

		tSpec, ok := typesSpecs[t]
		if !ok {
			panic("no specification for type " + t)
		}

		if _, ok := tSpec.Type.(*ast.ArrayType); ok {
			if !isValidErrorArrayTypeName(t) {
				reportAboutErrorType(pass, tSpec.Pos(), t, true)
			}
		} else if !isValidErrorTypeName(t) {
			reportAboutErrorType(pass, tSpec.Pos(), t, false)
		}
	})

	errorFuncs := stringSet{}
	insp.Preorder(funcs, func(node ast.Node) {
		f := node.(*ast.FuncDecl)
		if isFuncReturningErr(f.Type, allTypes, errorTypes) {
			errorFuncs[f.Name.Name] = struct{}{}
		}
	})

	inspectPkgLevelVarsOnly := func(node ast.Node) bool {
		switch v := node.(type) {
		case *ast.FuncDecl:
			return false

		case *ast.ValueSpec:
			if name, ok := isSentinelError(v, pkgAliases, allTypes, errorTypes, errorFuncs); ok && !isValidErrorVarName(name) {
				reportAboutErrorVar(pass, v.Pos(), name)
			}
		}
		return true
	}
	for _, f := range pass.Files {
		ast.Inspect(f, inspectPkgLevelVarsOnly)
	}

	return nil, nil
}

func reportAboutErrorType(pass *analysis.Pass, typePos token.Pos, typeName string, isArrayType bool) {
	var form string
	if unicode.IsLower([]rune(typeName)[0]) {
		form = "xxxError"
	} else {
		form = "XxxError"
	}

	if isArrayType {
		form += "s"
	}
	pass.Reportf(typePos, "the type name `%s` should conform to the `%s` format", typeName, form)
}

func reportAboutErrorVar(pass *analysis.Pass, pos token.Pos, varName string) {
	var form string
	if unicode.IsLower([]rune(varName)[0]) {
		form = "errXxx"
	} else {
		form = "ErrXxx"
	}
	pass.Reportf(pos, "the variable name `%s` should conform to the `%s` format", varName, form)
}

func getPkgFromPath(p string) string {
	idx := strings.LastIndex(p, "/")
	if idx == -1 {
		return p
	}
	return p[idx+1:]
}
