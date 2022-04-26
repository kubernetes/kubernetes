package analyzer

import (
	"go/ast"
	"go/token"
	"strings"
	"unicode"
)

func isMethodError(f *ast.FuncDecl) (typeName string, ok bool) {
	if f.Recv == nil {
		return "", false
	}
	if f.Name.Name != "Error" {
		return "", false
	}

	if f.Type == nil || f.Type.Results == nil || len(f.Type.Results.List) != 1 {
		return "", false
	}

	returnType, ok := f.Type.Results.List[0].Type.(*ast.Ident)
	if !ok {
		return "", false
	}

	var receiverType string

	switch rt := f.Recv.List[0].Type.(type) {
	case *ast.Ident:
		receiverType = rt.Name
	case *ast.StarExpr:
		if i, ok := rt.X.(*ast.Ident); ok {
			receiverType = i.Name
		}
	}

	return receiverType, returnType.Name == "string"
}

func isValidErrorTypeName(s string) bool {
	if isInitialism(s) {
		return true
	}

	words := split(s)
	wordsCnt := wordsCount(words)

	if wordsCnt["error"] != 1 {
		return false
	}
	return words[len(words)-1] == "error"
}

func isValidErrorArrayTypeName(s string) bool {
	if isInitialism(s) {
		return true
	}

	words := split(s)
	wordsCnt := wordsCount(words)

	if wordsCnt["errors"] != 1 {
		return false
	}
	return words[len(words)-1] == "errors"
}

func isFuncReturningErr(fType *ast.FuncType, allTypes, errorTypes stringSet) bool {
	if fType == nil || fType.Results == nil || len(fType.Results.List) != 1 {
		return false
	}

	var returnTypeName string
	switch rt := fType.Results.List[0].Type.(type) {
	case *ast.Ident:
		returnTypeName = rt.Name
	case *ast.StarExpr:
		if i, ok := rt.X.(*ast.Ident); ok {
			returnTypeName = i.Name
		}
	}

	return isErrorType(returnTypeName, allTypes, errorTypes)
}

func isErrorType(tName string, allTypes, errorTypes stringSet) bool {
	_, isUserType := allTypes[tName]
	_, isErrType := errorTypes[tName]
	return isErrType || (tName == "error" && !isUserType)
}

var knownErrConstructors = stringSet{
	"fmt.Errorf":           {},
	"errors.Errorf":        {},
	"errors.New":           {},
	"errors.Newf":          {},
	"errors.NewWithDepth":  {},
	"errors.NewWithDepthf": {},
	"errors.NewAssertionErrorWithWrappedErrf": {},
}

func isSentinelError( //nolint:gocognit
	v *ast.ValueSpec,
	pkgAliases map[string]string,
	allTypes, errorTypes, errorFuncs stringSet,
) (varName string, ok bool) {
	if len(v.Names) != 1 {
		return "", false
	}
	varName = v.Names[0].Name

	switch vv := v.Type.(type) {
	// var ErrEndOfFile error
	// var ErrEndOfFile SomeErrType
	case *ast.Ident:
		if isErrorType(vv.Name, allTypes, errorTypes) {
			return varName, true
		}

	// var ErrEndOfFile *SomeErrType
	case *ast.StarExpr:
		if i, ok := vv.X.(*ast.Ident); ok && isErrorType(i.Name, allTypes, errorTypes) {
			return varName, true
		}
	}

	if len(v.Values) != 1 {
		return "", false
	}

	switch vv := v.Values[0].(type) {
	case *ast.CallExpr:
		switch fun := vv.Fun.(type) {
		// var ErrEndOfFile = errors.New("end of file")
		case *ast.SelectorExpr:
			pkg, ok := fun.X.(*ast.Ident)
			if !ok {
				return "", false
			}
			pkgFun := fun.Sel

			pkgName := pkg.Name
			if a, ok := pkgAliases[pkgName]; ok {
				pkgName = a
			}

			_, ok = knownErrConstructors[pkgName+"."+pkgFun.Name]
			return varName, ok

		// var ErrEndOfFile = newErrEndOfFile()
		// var ErrEndOfFile = new(EndOfFileError)
		// const ErrEndOfFile = constError("end of file")
		case *ast.Ident:
			if isErrorType(fun.Name, allTypes, errorTypes) {
				return varName, true
			}

			if _, ok := errorFuncs[fun.Name]; ok {
				return varName, true
			}

			if fun.Name == "new" && len(vv.Args) == 1 {
				if i, ok := vv.Args[0].(*ast.Ident); ok {
					return varName, isErrorType(i.Name, allTypes, errorTypes)
				}
			}

		// var ErrEndOfFile = func() error { ... }
		case *ast.FuncLit:
			return varName, isFuncReturningErr(fun.Type, allTypes, errorTypes)
		}

	// var ErrEndOfFile = &EndOfFileError{}
	case *ast.UnaryExpr:
		if vv.Op == token.AND { // &
			if lit, ok := vv.X.(*ast.CompositeLit); ok {
				if i, ok := lit.Type.(*ast.Ident); ok {
					return varName, isErrorType(i.Name, allTypes, errorTypes)
				}
			}
		}

	// var ErrEndOfFile = EndOfFileError{}
	case *ast.CompositeLit:
		if i, ok := vv.Type.(*ast.Ident); ok {
			return varName, isErrorType(i.Name, allTypes, errorTypes)
		}
	}

	return "", false
}

func isValidErrorVarName(s string) bool {
	if isInitialism(s) {
		return true
	}

	words := split(s)
	wordsCnt := wordsCount(words)

	if wordsCnt["err"] != 1 {
		return false
	}
	return words[0] == "err"
}

func isInitialism(s string) bool {
	return strings.ToLower(s) == s || strings.ToUpper(s) == s
}

func split(s string) []string {
	var words []string
	ss := []rune(s)

	var b strings.Builder
	b.WriteRune(ss[0])

	for _, r := range ss[1:] {
		if unicode.IsUpper(r) {
			words = append(words, strings.ToLower(b.String()))
			b.Reset()
		}
		b.WriteRune(r)
	}

	words = append(words, strings.ToLower(b.String()))
	return words
}

func wordsCount(w []string) map[string]int {
	result := make(map[string]int, len(w))
	for _, ww := range w {
		result[ww]++
	}
	return result
}
