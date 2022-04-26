package halstvol

import (
	"go/ast"
	"math"
)

type HalstVol struct {
	Val  float64
	Coef Coef
}

type Coef struct {
	Opt map[string]int
	Opd map[string]int
}

func (v *HalstVol) Analyze(n ast.Node) {
	switch n := n.(type) {
	case *ast.FuncDecl, *ast.GenDecl:
		v.handleDecl(n)
	case *ast.ParenExpr, *ast.IndexExpr, *ast.SliceExpr, *ast.TypeAssertExpr, *ast.CallExpr, *ast.StarExpr, *ast.UnaryExpr, *ast.BinaryExpr, *ast.KeyValueExpr:
		v.handleExpr(n)
	case *ast.BasicLit, *ast.CompositeLit:
		v.handleLit(n)
	case *ast.Ident:
		v.handleIdent(n)
	case *ast.Ellipsis:
		incrIfAllTrue(v.Coef.Opt, "...", []bool{n.Ellipsis.IsValid()})
	case *ast.FuncType:
		incrIfAllTrue(v.Coef.Opt, "func", []bool{n.Func.IsValid()})
		v.Coef.Opt["()"]++
	case *ast.ChanType:
		incrIfAllTrue(v.Coef.Opt, "chan", []bool{n.Begin.IsValid()})
		incrIfAllTrue(v.Coef.Opt, "<-", []bool{n.Arrow.IsValid()})
	case *ast.SendStmt, *ast.IncDecStmt, *ast.AssignStmt, *ast.GoStmt, *ast.DeferStmt, *ast.ReturnStmt, *ast.BranchStmt, *ast.BlockStmt, *ast.IfStmt, *ast.SwitchStmt, *ast.SelectStmt, *ast.ForStmt, *ast.RangeStmt:
		v.handleStmt(n)
	case *ast.CaseClause:
		v.handleCaseClause(n)
	}
}

func (v *HalstVol) Calc() {
	distOpt := len(v.Coef.Opt)
	distOpd := len(v.Coef.Opd)

	var sumOpt, sumOpd int

	for _, val := range v.Coef.Opt {
		sumOpt += val
	}

	for _, val := range v.Coef.Opd {
		sumOpd += val
	}

	vocab := distOpt + distOpd
	length := sumOpt + sumOpd

	v.Val = float64(length) * math.Log2(float64(vocab))
}

// TODO: Consider the necessity
func incrIfAllTrue(coef map[string]int, sym string, cond []bool) {
	for _, ok := range cond {
		if !ok {
			return
		}
	}
	coef[sym]++
}
