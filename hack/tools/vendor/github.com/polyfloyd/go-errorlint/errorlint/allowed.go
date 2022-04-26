package errorlint

import (
	"fmt"
	"go/ast"
)

var allowedErrors = []struct {
	err string
	fun string
}{
	// pkg/archive/tar
	{err: "io.EOF", fun: "(*tar.Reader).Next"},
	{err: "io.EOF", fun: "(*tar.Reader).Read"},
	// pkg/bufio
	{err: "io.EOF", fun: "(*bufio.Reader).Discard"},
	{err: "io.EOF", fun: "(*bufio.Reader).Peek"},
	{err: "io.EOF", fun: "(*bufio.Reader).Read"},
	{err: "io.EOF", fun: "(*bufio.Reader).ReadByte"},
	{err: "io.EOF", fun: "(*bufio.Reader).ReadBytes"},
	{err: "io.EOF", fun: "(*bufio.Reader).ReadLine"},
	{err: "io.EOF", fun: "(*bufio.Reader).ReadSlice"},
	{err: "io.EOF", fun: "(*bufio.Reader).ReadString"},
	{err: "io.EOF", fun: "(*bufio.Scanner).Scan"},
	// pkg/bytes
	{err: "io.EOF", fun: "(*bytes.Buffer).Read"},
	{err: "io.EOF", fun: "(*bytes.Buffer).ReadByte"},
	{err: "io.EOF", fun: "(*bytes.Buffer).ReadBytes"},
	{err: "io.EOF", fun: "(*bytes.Buffer).ReadRune"},
	{err: "io.EOF", fun: "(*bytes.Buffer).ReadString"},
	{err: "io.EOF", fun: "(*bytes.Reader).Read"},
	{err: "io.EOF", fun: "(*bytes.Reader).ReadAt"},
	{err: "io.EOF", fun: "(*bytes.Reader).ReadByte"},
	{err: "io.EOF", fun: "(*bytes.Reader).ReadRune"},
	{err: "io.EOF", fun: "(*bytes.Reader).ReadString"},
	// pkg/database/sql
	{err: "sql.ErrNoRows", fun: "(*database/sql.Row).Scan"},
	// pkg/io
	{err: "io.EOF", fun: "(io.Reader).Read"},
	{err: "io.ErrClosedPipe", fun: "(*io.PipeWriter).Write"},
	{err: "io.ErrShortBuffer", fun: "io.ReadAtLeast"},
	{err: "io.ErrUnexpectedEOF", fun: "io.ReadAtLeast"},
	{err: "io.ErrUnexpectedEOF", fun: "io.ReadFull"},
	// pkg/net/http
	{err: "http.ErrServerClosed", fun: "(*net/http.Server).ListenAndServe"},
	{err: "http.ErrServerClosed", fun: "(*net/http.Server).ListenAndServeTLS"},
	{err: "http.ErrServerClosed", fun: "(*net/http.Server).Serve"},
	{err: "http.ErrServerClosed", fun: "(*net/http.Server).ServeTLS"},
	{err: "http.ErrServerClosed", fun: "http.ListenAndServe"},
	{err: "http.ErrServerClosed", fun: "http.ListenAndServeTLS"},
	{err: "http.ErrServerClosed", fun: "http.Serve"},
	{err: "http.ErrServerClosed", fun: "http.ServeTLS"},
	// pkg/os
	{err: "io.EOF", fun: "(*os.File).Read"},
	{err: "io.EOF", fun: "(*os.File).ReadAt"},
	{err: "io.EOF", fun: "(*os.File).ReadDir"},
	{err: "io.EOF", fun: "(*os.File).Readdir"},
	{err: "io.EOF", fun: "(*os.File).Readdirnames"},
	// pkg/strings
	{err: "io.EOF", fun: "(*strings.Reader).Read"},
	{err: "io.EOF", fun: "(*strings.Reader).ReadAt"},
	{err: "io.EOF", fun: "(*strings.Reader).ReadByte"},
	{err: "io.EOF", fun: "(*strings.Reader).ReadRune"},
}

func isAllowedErrAndFunc(err, fun string) bool {
	for _, allow := range allowedErrors {
		if allow.fun == fun && allow.err == err {
			return true
		}
	}
	return false
}

func isAllowedErrorComparison(info *TypesInfoExt, binExpr *ast.BinaryExpr) bool {
	var errName string // `<package>.<name>`, e.g. `io.EOF`
	var callExprs []*ast.CallExpr

	// Figure out which half of the expression is the returned error and which
	// half is the presumed error declaration.
	for _, expr := range []ast.Expr{binExpr.X, binExpr.Y} {
		switch t := expr.(type) {
		case *ast.SelectorExpr:
			// A selector which we assume refers to a staticaly declared error
			// in a package.
			errName = selectorToString(t)
		case *ast.Ident:
			// Identifier, most likely to be the `err` variable or whatever
			// produces it.
			callExprs = assigningCallExprs(info, t)
		case *ast.CallExpr:
			callExprs = append(callExprs, t)
		}
	}

	// Unimplemented or not sure, disallow the expression.
	if errName == "" || len(callExprs) == 0 {
		return false
	}

	// Map call expressions to the function name format of the allow list.
	functionNames := make([]string, len(callExprs))
	for i, callExpr := range callExprs {
		functionSelector, ok := callExpr.Fun.(*ast.SelectorExpr)
		if !ok {
			// If the function is not a selector it is not an Std function that is
			// allowed.
			return false
		}
		if sel, ok := info.Selections[functionSelector]; ok {
			functionNames[i] = fmt.Sprintf("(%s).%s", sel.Recv(), sel.Obj().Name())
		} else {
			// If there is no selection, assume it is a package.
			functionNames[i] = selectorToString(callExpr.Fun.(*ast.SelectorExpr))
		}
	}

	// All assignments done must be allowed.
	for _, funcName := range functionNames {
		if !isAllowedErrAndFunc(errName, funcName) {
			return false
		}
	}
	return true
}

// assigningCallExprs finds all *ast.CallExpr nodes that are part of an
// *ast.AssignStmt that assign to the subject identifier.
func assigningCallExprs(info *TypesInfoExt, subject *ast.Ident) []*ast.CallExpr {
	if subject.Obj == nil {
		return nil
	}

	// Find other identifiers that reference this same object. Make sure to
	// exclude the subject identifier as it will cause an infinite recursion
	// and is being used in a read operation anyway.
	sobj := info.ObjectOf(subject)
	identifiers := []*ast.Ident{}
	for _, ident := range info.IdentifiersForObject[sobj] {
		if subject.Pos() != ident.Pos() {
			identifiers = append(identifiers, ident)
		}
	}

	// Find out whether the identifiers are part of an assignment statement.
	var callExprs []*ast.CallExpr
	for _, ident := range identifiers {
		parent := info.NodeParent[ident]
		switch declT := parent.(type) {
		case *ast.AssignStmt:
			// The identifier is LHS of an assignment.
			assignment := declT

			assigningExpr := assignment.Rhs[0]
			// If the assignment is comprised of multiple expressions, find out
			// which LHS expression we should use by finding its index in the LHS.
			if len(assignment.Rhs) > 1 {
				for i, lhs := range assignment.Lhs {
					if subject.Name == lhs.(*ast.Ident).Name {
						assigningExpr = assignment.Rhs[i]
						break
					}
				}
			}

			switch assignT := assigningExpr.(type) {
			case *ast.CallExpr:
				// Found the function call.
				callExprs = append(callExprs, assignT)
			case *ast.Ident:
				// The subject was the result of assigning from another identifier.
				callExprs = append(callExprs, assigningCallExprs(info, assignT)...)
			default:
				// TODO: inconclusive?
			}
		}
	}
	return callExprs
}

func selectorToString(selExpr *ast.SelectorExpr) string {
	if ident, ok := selExpr.X.(*ast.Ident); ok {
		return ident.Name + "." + selExpr.Sel.Name
	}
	return ""
}
