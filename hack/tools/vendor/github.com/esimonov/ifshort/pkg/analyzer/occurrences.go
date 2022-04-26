package analyzer

import (
	"go/ast"
	"go/token"
	"time"

	"golang.org/x/tools/go/analysis"
)

// occurrence is a variable occurrence.
type occurrence struct {
	declarationPos token.Pos
	ifStmtPos      token.Pos
}

func (occ *occurrence) isComplete() bool {
	return occ.ifStmtPos != token.NoPos && occ.declarationPos != token.NoPos
}

// scopeMarkeredOccurences is a map of scope markers to variable occurrences.
type scopeMarkeredOccurences map[int64]occurrence

func (smo scopeMarkeredOccurences) getGreatestMarker() int64 {
	var maxScopeMarker int64

	for marker := range smo {
		if marker > maxScopeMarker {
			maxScopeMarker = marker
		}
	}
	return maxScopeMarker
}

// find scope marker of the greatest token.Pos that is smaller than provided.
func (smo scopeMarkeredOccurences) getScopeMarkerForPosition(pos token.Pos) int64 {
	var m int64
	var foundPos token.Pos

	for marker, occ := range smo {
		if occ.declarationPos < pos && occ.declarationPos >= foundPos {
			m = marker
			foundPos = occ.declarationPos
		}
	}
	return m
}

func (smo scopeMarkeredOccurences) isEmponymousKey(pos token.Pos) bool {
	if pos == token.NoPos {
		return false
	}

	for _, occ := range smo {
		if occ.ifStmtPos == pos {
			return true
		}
	}
	return false
}

// namedOccurrenceMap is a map of variable names to scopeMarkeredOccurences.
type namedOccurrenceMap map[string]scopeMarkeredOccurences

func getNamedOccurrenceMap(fdecl *ast.FuncDecl, pass *analysis.Pass) namedOccurrenceMap {
	nom := namedOccurrenceMap(map[string]scopeMarkeredOccurences{})

	if fdecl == nil || fdecl.Body == nil {
		return nom
	}

	for _, stmt := range fdecl.Body.List {
		switch v := stmt.(type) {
		case *ast.AssignStmt:
			nom.addFromAssignment(pass, v)
		case *ast.IfStmt:
			nom.addFromCondition(v)
			nom.addFromIfClause(v)
			nom.addFromElseClause(v)
		}
	}

	candidates := namedOccurrenceMap(map[string]scopeMarkeredOccurences{})

	for varName, markeredOccs := range nom {
		for marker, occ := range markeredOccs {
			if !occ.isComplete() && !nom.isFoundByScopeMarker(marker) {
				continue
			}
			if _, ok := candidates[varName]; !ok {
				candidates[varName] = scopeMarkeredOccurences{
					marker: occ,
				}
			} else {
				candidates[varName][marker] = occ
			}
		}
	}
	return candidates
}

func (nom namedOccurrenceMap) isFoundByScopeMarker(scopeMarker int64) bool {
	var i int

	for _, markeredOccs := range nom {
		for marker := range markeredOccs {
			if marker == scopeMarker {
				i++
			}
		}
	}
	return i >= 2
}

func (nom namedOccurrenceMap) addFromAssignment(pass *analysis.Pass, assignment *ast.AssignStmt) {
	if assignment.Tok != token.DEFINE {
		return
	}

	scopeMarker := time.Now().UnixNano()

	for i, el := range assignment.Lhs {
		ident, ok := el.(*ast.Ident)
		if !ok {
			continue
		}

		if ident.Name == "_" || ident.Obj == nil || isUnshortenableAssignment(ident.Obj.Decl) {
			continue
		}

		if markeredOccs, ok := nom[ident.Name]; ok {
			markeredOccs[scopeMarker] = occurrence{
				declarationPos: ident.Pos(),
			}
			nom[ident.Name] = markeredOccs
		} else {
			newOcc := occurrence{}
			if areFlagSettingsSatisfied(pass, assignment, i) {
				newOcc.declarationPos = ident.Pos()
			}
			nom[ident.Name] = scopeMarkeredOccurences{scopeMarker: newOcc}
		}
	}
}

func isUnshortenableAssignment(decl interface{}) bool {
	assign, ok := decl.(*ast.AssignStmt)
	if !ok {
		return false
	}

	for _, el := range assign.Rhs {
		u, ok := el.(*ast.UnaryExpr)
		if !ok {
			continue
		}

		if u.Op == token.AND {
			if _, ok := u.X.(*ast.CompositeLit); ok {
				return true
			}
		}
	}
	return false
}

func areFlagSettingsSatisfied(pass *analysis.Pass, assignment *ast.AssignStmt, i int) bool {
	lh := assignment.Lhs[i]
	rh := assignment.Rhs[len(assignment.Rhs)-1]

	if len(assignment.Rhs) == len(assignment.Lhs) {
		rh = assignment.Rhs[i]
	}

	if pass.Fset.Position(rh.End()).Line-pass.Fset.Position(rh.Pos()).Line > maxDeclLines {
		return false
	}
	if int(rh.End()-lh.Pos()) > maxDeclChars {
		return false
	}
	return true
}

func (nom namedOccurrenceMap) addFromCondition(stmt *ast.IfStmt) {
	switch v := stmt.Cond.(type) {
	case *ast.BinaryExpr:
		for _, v := range [2]ast.Expr{v.X, v.Y} {
			switch e := v.(type) {
			case *ast.CallExpr:
				nom.addFromCallExpr(stmt.If, e)
			case *ast.Ident:
				nom.addFromIdent(stmt.If, e)
			case *ast.SelectorExpr:
				nom.addFromIdent(stmt.If, e.X)
			}
		}
	case *ast.CallExpr:
		for _, a := range v.Args {
			switch e := a.(type) {
			case *ast.Ident:
				nom.addFromIdent(stmt.If, e)
			case *ast.CallExpr:
				nom.addFromCallExpr(stmt.If, e)
			}
		}
	case *ast.Ident:
		nom.addFromIdent(stmt.If, v)
	case *ast.UnaryExpr:
		switch e := v.X.(type) {
		case *ast.Ident:
			nom.addFromIdent(stmt.If, e)
		case *ast.SelectorExpr:
			nom.addFromIdent(stmt.If, e.X)
		}
	}
}

func (nom namedOccurrenceMap) addFromIfClause(stmt *ast.IfStmt) {
	nom.addFromBlockStmt(stmt.Body, stmt.If)
}

func (nom namedOccurrenceMap) addFromElseClause(stmt *ast.IfStmt) {
	nom.addFromBlockStmt(stmt.Else, stmt.If)
}

func (nom namedOccurrenceMap) addFromBlockStmt(stmt ast.Stmt, ifPos token.Pos) {
	blockStmt, ok := stmt.(*ast.BlockStmt)
	if !ok {
		return
	}

	for _, el := range blockStmt.List {
		exptStmt, ok := el.(*ast.ExprStmt)
		if !ok {
			continue
		}

		if callExpr, ok := exptStmt.X.(*ast.CallExpr); ok {
			nom.addFromCallExpr(ifPos, callExpr)
		}
	}
}

func (nom namedOccurrenceMap) addFromCallExpr(ifPos token.Pos, callExpr *ast.CallExpr) {
	for _, arg := range callExpr.Args {
		nom.addFromIdent(ifPos, arg)
	}
}

func (nom namedOccurrenceMap) addFromIdent(ifPos token.Pos, v ast.Expr) {
	ident, ok := v.(*ast.Ident)
	if !ok {
		return
	}

	if markeredOccs, ok := nom[ident.Name]; ok {
		marker := nom[ident.Name].getGreatestMarker()

		occ := markeredOccs[marker]
		if occ.isComplete() {
			return
		}

		occ.ifStmtPos = ifPos
		nom[ident.Name][marker] = occ
	}
}
