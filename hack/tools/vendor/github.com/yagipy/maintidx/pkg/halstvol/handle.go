package halstvol

import "go/ast"

func (v *HalstVol) handleDecl(decl ast.Node) {
	switch n := decl.(type) {
	case *ast.FuncDecl:
		if n.Recv == nil {
			// In the case of receiver functions, the function name is incremented in *ast.Ident
			v.Coef.Opt[n.Name.Name]++
		} else {
			v.Coef.Opt["()"]++
		}
	case *ast.GenDecl:
		if n.Lparen.IsValid() && n.Rparen.IsValid() {
			v.Coef.Opt["()"]++
		}

		if n.Tok.IsOperator() {
			v.Coef.Opt[n.Tok.String()]++
		} else {
			v.Coef.Opd[n.Tok.String()]++
		}
	}
}

func (v *HalstVol) handleIdent(ident *ast.Ident) {
	if ident.Obj == nil {
		v.Coef.Opt[ident.Name]++
	} else {
		if ident.Obj.Kind.String() != "func" {
			v.Coef.Opd[ident.Name]++
		}
	}
}

func (v *HalstVol) handleLit(lit ast.Node) {
	switch n := lit.(type) {
	case *ast.BasicLit:
		if n.Kind.IsLiteral() {
			v.Coef.Opd[n.Value]++
		} else {
			v.Coef.Opt[n.Value]++
		}
	case *ast.CompositeLit:
		incrIfAllTrue(v.Coef.Opt, "{}", []bool{n.Lbrace.IsValid(), n.Rbrace.IsValid()})
	}
}

func (v *HalstVol) handleExpr(expr ast.Node) {
	switch n := expr.(type) {
	case *ast.ParenExpr:
		incrIfAllTrue(v.Coef.Opt, "()", []bool{n.Lparen.IsValid(), n.Rparen.IsValid()})
	case *ast.IndexExpr:
		incrIfAllTrue(v.Coef.Opt, "{}", []bool{n.Lbrack.IsValid(), n.Rbrack.IsValid()})
	case *ast.SliceExpr:
		incrIfAllTrue(v.Coef.Opt, "[]", []bool{n.Lbrack.IsValid(), n.Rbrack.IsValid()})
	case *ast.TypeAssertExpr:
		incrIfAllTrue(v.Coef.Opt, "()", []bool{n.Lparen.IsValid(), n.Rparen.IsValid()})
	case *ast.CallExpr:
		incrIfAllTrue(v.Coef.Opt, "()", []bool{n.Lparen.IsValid(), n.Rparen.IsValid()})
		incrIfAllTrue(v.Coef.Opt, "...", []bool{n.Ellipsis != 0})
	case *ast.StarExpr:
		incrIfAllTrue(v.Coef.Opt, "*", []bool{n.Star.IsValid()})
	case *ast.UnaryExpr:
		if n.Op.IsOperator() {
			v.Coef.Opt[n.Op.String()]++
		} else {
			v.Coef.Opd[n.Op.String()]++
		}
	case *ast.BinaryExpr:
		v.Coef.Opt[n.Op.String()]++
	case *ast.KeyValueExpr:
		incrIfAllTrue(v.Coef.Opt, ":", []bool{n.Colon.IsValid()})
	}
}

func (v *HalstVol) handleStmt(stmt ast.Node) {
	switch n := stmt.(type) {
	case *ast.SendStmt:
		incrIfAllTrue(v.Coef.Opt, "<-", []bool{n.Arrow.IsValid()})
	case *ast.IncDecStmt:
		incrIfAllTrue(v.Coef.Opt, n.Tok.String(), []bool{n.Tok.IsOperator()})
	case *ast.AssignStmt:
		if n.Tok.IsOperator() {
			v.Coef.Opt[n.Tok.String()]++
		}
	case *ast.GoStmt:
		if n.Go.IsValid() {
			v.Coef.Opt["go"]++
		}
	case *ast.DeferStmt:
		if n.Defer.IsValid() {
			v.Coef.Opt["defer"]++
		}
	case *ast.ReturnStmt:
		if n.Return.IsValid() {
			v.Coef.Opt["return"]++
		}
	case *ast.BranchStmt:
		if n.Tok.IsOperator() {
			v.Coef.Opt[n.Tok.String()]++
		} else {
			v.Coef.Opd[n.Tok.String()]++
		}
	case *ast.BlockStmt:
		if n.Lbrace.IsValid() && n.Rbrace.IsValid() {
			v.Coef.Opt["{}"]++
		}
	case *ast.IfStmt:
		if n.If.IsValid() {
			v.Coef.Opt["if"]++
		}
		if n.Else != nil {
			v.Coef.Opt["else"]++
		}
	case *ast.SwitchStmt:
		if n.Switch.IsValid() {
			v.Coef.Opt["switch"]++
		}
	case *ast.SelectStmt:
		if n.Select.IsValid() {
			v.Coef.Opt["select"]++
		}
	case *ast.ForStmt:
		if n.For.IsValid() {
			v.Coef.Opt["for"]++
		}
	case *ast.RangeStmt:
		if n.For.IsValid() {
			v.Coef.Opt["for"]++
		}
		if n.Key != nil {
			if n.Tok.IsOperator() {
				v.Coef.Opt[n.Tok.String()]++
			} else {
				v.Coef.Opd[n.Tok.String()]++
			}
		}
		v.Coef.Opt["range"]++
	}
}

func (v *HalstVol) handleCaseClause(cc *ast.CaseClause) {
	if cc.List == nil {
		v.Coef.Opt["default"]++
	}
	if cc.Colon.IsValid() {
		v.Coef.Opt[":"]++
	}
}
