package gogrep

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strconv"

	"github.com/go-toolsmith/astequal"
)

type matcher struct {
	prog *program

	insts []instruction
}

func newMatcher(prog *program) *matcher {
	return &matcher{
		prog:  prog,
		insts: prog.insts,
	}
}

func (m *matcher) nextInst(state *MatcherState) instruction {
	inst := m.insts[state.pc]
	state.pc++
	return inst
}

func (m *matcher) stringValue(inst instruction) string {
	return m.prog.strings[inst.valueIndex]
}

func (m *matcher) ifaceValue(inst instruction) interface{} {
	return m.prog.ifaces[inst.valueIndex]
}

func (m *matcher) MatchNode(state *MatcherState, n ast.Node, accept func(MatchData)) {
	state.pc = 0
	inst := m.nextInst(state)
	switch inst.op {
	case opMultiStmt:
		switch n := n.(type) {
		case *ast.BlockStmt:
			m.walkStmtSlice(state, n.List, accept)
		case *ast.CaseClause:
			m.walkStmtSlice(state, n.Body, accept)
		case *ast.CommClause:
			m.walkStmtSlice(state, n.Body, accept)
		}
	case opMultiExpr:
		switch n := n.(type) {
		case *ast.CallExpr:
			m.walkExprSlice(state, n.Args, accept)
		case *ast.CompositeLit:
			m.walkExprSlice(state, n.Elts, accept)
		case *ast.ReturnStmt:
			m.walkExprSlice(state, n.Results, accept)
		}
	case opMultiDecl:
		if n, ok := n.(*ast.File); ok {
			m.walkDeclSlice(state, n.Decls, accept)
		}
	default:
		state.capture = state.capture[:0]
		if m.matchNodeWithInst(state, inst, n) {
			accept(MatchData{
				Capture: state.capture,
				Node:    n,
			})
		}
	}
}

func (m *matcher) walkDeclSlice(state *MatcherState, decls []ast.Decl, accept func(MatchData)) {
	m.walkNodeSlice(state, declSlice(decls), accept)
}

func (m *matcher) walkExprSlice(state *MatcherState, exprs []ast.Expr, accept func(MatchData)) {
	m.walkNodeSlice(state, ExprSlice(exprs), accept)
}

func (m *matcher) walkStmtSlice(state *MatcherState, stmts []ast.Stmt, accept func(MatchData)) {
	m.walkNodeSlice(state, stmtSlice(stmts), accept)
}

func (m *matcher) walkNodeSlice(state *MatcherState, nodes NodeSlice, accept func(MatchData)) {
	sliceLen := nodes.Len()
	from := 0
	for {
		state.pc = 1 // FIXME: this is a kludge
		state.capture = state.capture[:0]
		matched, offset := m.matchNodeList(state, nodes.slice(from, sliceLen), true)
		if matched == nil {
			break
		}
		accept(MatchData{
			Capture: state.capture,
			Node:    matched,
		})
		from += offset - 1
		if from >= sliceLen {
			break
		}
	}
}

func (m *matcher) matchNamed(state *MatcherState, name string, n ast.Node) bool {
	prev, ok := findNamed(state.capture, name)
	if !ok {
		// First occurrence, record value.
		state.capture = append(state.capture, CapturedNode{Name: name, Node: n})
		return true
	}
	return equalNodes(prev, n)
}

func (m *matcher) matchNamedField(state *MatcherState, name string, n ast.Node) bool {
	prev, ok := findNamed(state.capture, name)
	if !ok {
		// First occurrence, record value.
		unwrapped := m.unwrapNode(n)
		state.capture = append(state.capture, CapturedNode{Name: name, Node: unwrapped})
		return true
	}
	n = m.unwrapNode(n)
	return equalNodes(prev, n)
}

func (m *matcher) unwrapNode(x ast.Node) ast.Node {
	switch x := x.(type) {
	case *ast.Field:
		if len(x.Names) == 0 {
			return x.Type
		}
	case *ast.FieldList:
		if x != nil && len(x.List) == 1 && len(x.List[0].Names) == 0 {
			return x.List[0].Type
		}
	}
	return x
}

func (m *matcher) matchNodeWithInst(state *MatcherState, inst instruction, n ast.Node) bool {
	switch inst.op {
	case opNode:
		return n != nil
	case opOptNode:
		return true

	case opNamedNode:
		return n != nil && m.matchNamed(state, m.stringValue(inst), n)
	case opNamedOptNode:
		return m.matchNamed(state, m.stringValue(inst), n)

	case opFieldNode:
		n, ok := n.(*ast.FieldList)
		return ok && n != nil && len(n.List) == 1 && len(n.List[0].Names) == 0
	case opNamedFieldNode:
		return n != nil && m.matchNamedField(state, m.stringValue(inst), n)

	case opBasicLit:
		n, ok := n.(*ast.BasicLit)
		return ok && m.ifaceValue(inst) == literalValue(n)

	case opStrictIntLit:
		n, ok := n.(*ast.BasicLit)
		return ok && n.Kind == token.INT && m.stringValue(inst) == n.Value
	case opStrictFloatLit:
		n, ok := n.(*ast.BasicLit)
		return ok && n.Kind == token.FLOAT && m.stringValue(inst) == n.Value
	case opStrictCharLit:
		n, ok := n.(*ast.BasicLit)
		return ok && n.Kind == token.CHAR && m.stringValue(inst) == n.Value
	case opStrictStringLit:
		n, ok := n.(*ast.BasicLit)
		return ok && n.Kind == token.STRING && m.stringValue(inst) == n.Value
	case opStrictComplexLit:
		n, ok := n.(*ast.BasicLit)
		return ok && n.Kind == token.IMAG && m.stringValue(inst) == n.Value

	case opIdent:
		n, ok := n.(*ast.Ident)
		return ok && m.stringValue(inst) == n.Name

	case opPkg:
		n, ok := n.(*ast.Ident)
		if !ok {
			return false
		}
		obj := state.Types.ObjectOf(n)
		if obj == nil {
			return false
		}
		pkgName, ok := obj.(*types.PkgName)
		return ok && pkgName.Imported().Path() == m.stringValue(inst)

	case opBinaryExpr:
		n, ok := n.(*ast.BinaryExpr)
		return ok && n.Op == token.Token(inst.value) &&
			m.matchNode(state, n.X) && m.matchNode(state, n.Y)

	case opUnaryExpr:
		n, ok := n.(*ast.UnaryExpr)
		return ok && n.Op == token.Token(inst.value) && m.matchNode(state, n.X)

	case opStarExpr:
		n, ok := n.(*ast.StarExpr)
		return ok && m.matchNode(state, n.X)

	case opVariadicCallExpr:
		n, ok := n.(*ast.CallExpr)
		return ok && n.Ellipsis.IsValid() && m.matchNode(state, n.Fun) && m.matchArgList(state, n.Args)
	case opNonVariadicCallExpr:
		n, ok := n.(*ast.CallExpr)
		return ok && !n.Ellipsis.IsValid() && m.matchNode(state, n.Fun) && m.matchArgList(state, n.Args)
	case opCallExpr:
		n, ok := n.(*ast.CallExpr)
		return ok && m.matchNode(state, n.Fun) && m.matchArgList(state, n.Args)

	case opSimpleSelectorExpr:
		n, ok := n.(*ast.SelectorExpr)
		return ok && m.stringValue(inst) == n.Sel.Name && m.matchNode(state, n.X)
	case opSelectorExpr:
		n, ok := n.(*ast.SelectorExpr)
		return ok && m.matchNode(state, n.Sel) && m.matchNode(state, n.X)

	case opTypeAssertExpr:
		n, ok := n.(*ast.TypeAssertExpr)
		return ok && m.matchNode(state, n.X) && m.matchNode(state, n.Type)
	case opTypeSwitchAssertExpr:
		n, ok := n.(*ast.TypeAssertExpr)
		return ok && n.Type == nil && m.matchNode(state, n.X)

	case opSliceExpr:
		n, ok := n.(*ast.SliceExpr)
		return ok && n.Low == nil && n.High == nil && m.matchNode(state, n.X)
	case opSliceFromExpr:
		n, ok := n.(*ast.SliceExpr)
		return ok && n.High == nil && !n.Slice3 &&
			m.matchNode(state, n.X) && m.matchNode(state, n.Low)
	case opSliceToExpr:
		n, ok := n.(*ast.SliceExpr)
		return ok && n.Low == nil && !n.Slice3 &&
			m.matchNode(state, n.X) && m.matchNode(state, n.High)
	case opSliceFromToExpr:
		n, ok := n.(*ast.SliceExpr)
		return ok && !n.Slice3 &&
			m.matchNode(state, n.X) && m.matchNode(state, n.Low) && m.matchNode(state, n.High)
	case opSliceToCapExpr:
		n, ok := n.(*ast.SliceExpr)
		return ok && n.Low == nil &&
			m.matchNode(state, n.X) && m.matchNode(state, n.High) && m.matchNode(state, n.Max)
	case opSliceFromToCapExpr:
		n, ok := n.(*ast.SliceExpr)
		return ok && m.matchNode(state, n.X) && m.matchNode(state, n.Low) && m.matchNode(state, n.High) && m.matchNode(state, n.Max)

	case opIndexExpr:
		n, ok := n.(*ast.IndexExpr)
		return ok && m.matchNode(state, n.X) && m.matchNode(state, n.Index)

	case opKeyValueExpr:
		n, ok := n.(*ast.KeyValueExpr)
		return ok && m.matchNode(state, n.Key) && m.matchNode(state, n.Value)

	case opParenExpr:
		n, ok := n.(*ast.ParenExpr)
		return ok && m.matchNode(state, n.X)

	case opEllipsis:
		n, ok := n.(*ast.Ellipsis)
		return ok && n.Elt == nil
	case opTypedEllipsis:
		n, ok := n.(*ast.Ellipsis)
		return ok && n.Elt != nil && m.matchNode(state, n.Elt)

	case opSliceType:
		n, ok := n.(*ast.ArrayType)
		return ok && n.Len == nil && m.matchNode(state, n.Elt)
	case opArrayType:
		n, ok := n.(*ast.ArrayType)
		return ok && n.Len != nil && m.matchNode(state, n.Len) && m.matchNode(state, n.Elt)
	case opMapType:
		n, ok := n.(*ast.MapType)
		return ok && m.matchNode(state, n.Key) && m.matchNode(state, n.Value)
	case opChanType:
		n, ok := n.(*ast.ChanType)
		return ok && ast.ChanDir(inst.value) == n.Dir && m.matchNode(state, n.Value)
	case opVoidFuncType:
		n, ok := n.(*ast.FuncType)
		return ok && n.Results == nil && m.matchNode(state, n.Params)
	case opFuncType:
		n, ok := n.(*ast.FuncType)
		return ok && m.matchNode(state, n.Params) && m.matchNode(state, n.Results)
	case opStructType:
		n, ok := n.(*ast.StructType)
		return ok && m.matchNode(state, n.Fields)
	case opInterfaceType:
		n, ok := n.(*ast.InterfaceType)
		return ok && m.matchNode(state, n.Methods)

	case opCompositeLit:
		n, ok := n.(*ast.CompositeLit)
		return ok && n.Type == nil && m.matchExprSlice(state, n.Elts)
	case opTypedCompositeLit:
		n, ok := n.(*ast.CompositeLit)
		return ok && n.Type != nil && m.matchNode(state, n.Type) && m.matchExprSlice(state, n.Elts)

	case opUnnamedField:
		n, ok := n.(*ast.Field)
		return ok && len(n.Names) == 0 && m.matchNode(state, n.Type)
	case opSimpleField:
		n, ok := n.(*ast.Field)
		return ok && len(n.Names) == 1 && m.stringValue(inst) == n.Names[0].Name && m.matchNode(state, n.Type)
	case opField:
		n, ok := n.(*ast.Field)
		return ok && len(n.Names) == 1 && m.matchNode(state, n.Names[0]) && m.matchNode(state, n.Type)
	case opMultiField:
		n, ok := n.(*ast.Field)
		return ok && len(n.Names) >= 2 && m.matchIdentSlice(state, n.Names) && m.matchNode(state, n.Type)
	case opFieldList:
		// FieldList could be nil in places like function return types.
		n, ok := n.(*ast.FieldList)
		return ok && n != nil && m.matchFieldSlice(state, n.List)

	case opFuncLit:
		n, ok := n.(*ast.FuncLit)
		return ok && m.matchNode(state, n.Type) && m.matchNode(state, n.Body)

	case opAssignStmt:
		n, ok := n.(*ast.AssignStmt)
		return ok && token.Token(inst.value) == n.Tok &&
			len(n.Lhs) == 1 && m.matchNode(state, n.Lhs[0]) &&
			len(n.Rhs) == 1 && m.matchNode(state, n.Rhs[0])
	case opMultiAssignStmt:
		n, ok := n.(*ast.AssignStmt)
		return ok && token.Token(inst.value) == n.Tok &&
			m.matchExprSlice(state, n.Lhs) && m.matchExprSlice(state, n.Rhs)

	case opExprStmt:
		n, ok := n.(*ast.ExprStmt)
		return ok && m.matchNode(state, n.X)

	case opGoStmt:
		n, ok := n.(*ast.GoStmt)
		return ok && m.matchNode(state, n.Call)
	case opDeferStmt:
		n, ok := n.(*ast.DeferStmt)
		return ok && m.matchNode(state, n.Call)
	case opSendStmt:
		n, ok := n.(*ast.SendStmt)
		return ok && m.matchNode(state, n.Chan) && m.matchNode(state, n.Value)

	case opBlockStmt:
		n, ok := n.(*ast.BlockStmt)
		return ok && m.matchStmtSlice(state, n.List)

	case opIfStmt:
		n, ok := n.(*ast.IfStmt)
		return ok && n.Init == nil && n.Else == nil &&
			m.matchNode(state, n.Cond) && m.matchNode(state, n.Body)
	case opIfElseStmt:
		n, ok := n.(*ast.IfStmt)
		return ok && n.Init == nil && n.Else != nil &&
			m.matchNode(state, n.Cond) && m.matchNode(state, n.Body) && m.matchNode(state, n.Else)
	case opIfInitStmt:
		n, ok := n.(*ast.IfStmt)
		return ok && n.Else == nil &&
			m.matchNode(state, n.Init) && m.matchNode(state, n.Cond) && m.matchNode(state, n.Body)
	case opIfInitElseStmt:
		n, ok := n.(*ast.IfStmt)
		return ok && n.Else != nil &&
			m.matchNode(state, n.Init) && m.matchNode(state, n.Cond) && m.matchNode(state, n.Body) && m.matchNode(state, n.Else)

	case opIfNamedOptStmt:
		n, ok := n.(*ast.IfStmt)
		return ok && n.Else == nil && m.matchNode(state, n.Body) &&
			m.matchNamed(state, m.stringValue(inst), toStmtSlice(n.Cond, n.Init))
	case opIfNamedOptElseStmt:
		n, ok := n.(*ast.IfStmt)
		return ok && n.Else != nil && m.matchNode(state, n.Body) && m.matchNode(state, n.Else) &&
			m.matchNamed(state, m.stringValue(inst), toStmtSlice(n.Cond, n.Init))

	case opCaseClause:
		n, ok := n.(*ast.CaseClause)
		return ok && n.List != nil && m.matchExprSlice(state, n.List) && m.matchStmtSlice(state, n.Body)
	case opDefaultCaseClause:
		n, ok := n.(*ast.CaseClause)
		return ok && n.List == nil && m.matchStmtSlice(state, n.Body)

	case opSwitchStmt:
		n, ok := n.(*ast.SwitchStmt)
		return ok && n.Init == nil && n.Tag == nil && m.matchStmtSlice(state, n.Body.List)
	case opSwitchTagStmt:
		n, ok := n.(*ast.SwitchStmt)
		return ok && n.Init == nil && m.matchNode(state, n.Tag) && m.matchStmtSlice(state, n.Body.List)
	case opSwitchInitStmt:
		n, ok := n.(*ast.SwitchStmt)
		return ok && n.Tag == nil && m.matchNode(state, n.Init) && m.matchStmtSlice(state, n.Body.List)
	case opSwitchInitTagStmt:
		n, ok := n.(*ast.SwitchStmt)
		return ok && m.matchNode(state, n.Init) && m.matchNode(state, n.Tag) && m.matchStmtSlice(state, n.Body.List)

	case opTypeSwitchStmt:
		n, ok := n.(*ast.TypeSwitchStmt)
		return ok && n.Init == nil && m.matchNode(state, n.Assign) && m.matchStmtSlice(state, n.Body.List)
	case opTypeSwitchInitStmt:
		n, ok := n.(*ast.TypeSwitchStmt)
		return ok && m.matchNode(state, n.Init) &&
			m.matchNode(state, n.Assign) && m.matchStmtSlice(state, n.Body.List)

	case opCommClause:
		n, ok := n.(*ast.CommClause)
		return ok && n.Comm != nil && m.matchNode(state, n.Comm) && m.matchStmtSlice(state, n.Body)
	case opDefaultCommClause:
		n, ok := n.(*ast.CommClause)
		return ok && n.Comm == nil && m.matchStmtSlice(state, n.Body)

	case opSelectStmt:
		n, ok := n.(*ast.SelectStmt)
		return ok && m.matchStmtSlice(state, n.Body.List)

	case opRangeStmt:
		n, ok := n.(*ast.RangeStmt)
		return ok && n.Key == nil && n.Value == nil && m.matchNode(state, n.X) && m.matchNode(state, n.Body)
	case opRangeKeyStmt:
		n, ok := n.(*ast.RangeStmt)
		return ok && n.Key != nil && n.Value == nil && token.Token(inst.value) == n.Tok &&
			m.matchNode(state, n.Key) && m.matchNode(state, n.X) && m.matchNode(state, n.Body)
	case opRangeKeyValueStmt:
		n, ok := n.(*ast.RangeStmt)
		return ok && n.Key != nil && n.Value != nil && token.Token(inst.value) == n.Tok &&
			m.matchNode(state, n.Key) && m.matchNode(state, n.Value) && m.matchNode(state, n.X) && m.matchNode(state, n.Body)

	case opForStmt:
		n, ok := n.(*ast.ForStmt)
		return ok && n.Init == nil && n.Cond == nil && n.Post == nil &&
			m.matchNode(state, n.Body)
	case opForPostStmt:
		n, ok := n.(*ast.ForStmt)
		return ok && n.Init == nil && n.Cond == nil && n.Post != nil &&
			m.matchNode(state, n.Post) && m.matchNode(state, n.Body)
	case opForCondStmt:
		n, ok := n.(*ast.ForStmt)
		return ok && n.Init == nil && n.Cond != nil && n.Post == nil &&
			m.matchNode(state, n.Cond) && m.matchNode(state, n.Body)
	case opForCondPostStmt:
		n, ok := n.(*ast.ForStmt)
		return ok && n.Init == nil && n.Cond != nil && n.Post != nil &&
			m.matchNode(state, n.Cond) && m.matchNode(state, n.Post) && m.matchNode(state, n.Body)
	case opForInitStmt:
		n, ok := n.(*ast.ForStmt)
		return ok && n.Init != nil && n.Cond == nil && n.Post == nil &&
			m.matchNode(state, n.Init) && m.matchNode(state, n.Body)
	case opForInitPostStmt:
		n, ok := n.(*ast.ForStmt)
		return ok && n.Init != nil && n.Cond == nil && n.Post != nil &&
			m.matchNode(state, n.Init) && m.matchNode(state, n.Post) && m.matchNode(state, n.Body)
	case opForInitCondStmt:
		n, ok := n.(*ast.ForStmt)
		return ok && n.Init != nil && n.Cond != nil && n.Post == nil &&
			m.matchNode(state, n.Init) && m.matchNode(state, n.Cond) && m.matchNode(state, n.Body)
	case opForInitCondPostStmt:
		n, ok := n.(*ast.ForStmt)
		return ok && m.matchNode(state, n.Init) && m.matchNode(state, n.Cond) && m.matchNode(state, n.Post) && m.matchNode(state, n.Body)

	case opIncDecStmt:
		n, ok := n.(*ast.IncDecStmt)
		return ok && token.Token(inst.value) == n.Tok && m.matchNode(state, n.X)

	case opReturnStmt:
		n, ok := n.(*ast.ReturnStmt)
		return ok && m.matchExprSlice(state, n.Results)

	case opLabeledStmt:
		n, ok := n.(*ast.LabeledStmt)
		return ok && m.matchNode(state, n.Label) && m.matchNode(state, n.Stmt)
	case opSimpleLabeledStmt:
		n, ok := n.(*ast.LabeledStmt)
		return ok && m.stringValue(inst) == n.Label.Name && m.matchNode(state, n.Stmt)

	case opLabeledBranchStmt:
		n, ok := n.(*ast.BranchStmt)
		return ok && n.Label != nil && token.Token(inst.value) == n.Tok && m.matchNode(state, n.Label)
	case opSimpleLabeledBranchStmt:
		n, ok := n.(*ast.BranchStmt)
		return ok && n.Label != nil && m.stringValue(inst) == n.Label.Name && token.Token(inst.value) == n.Tok
	case opBranchStmt:
		n, ok := n.(*ast.BranchStmt)
		return ok && n.Label == nil && token.Token(inst.value) == n.Tok

	case opEmptyStmt:
		_, ok := n.(*ast.EmptyStmt)
		return ok

	case opFuncDecl:
		n, ok := n.(*ast.FuncDecl)
		return ok && n.Recv == nil && n.Body != nil &&
			m.matchNode(state, n.Name) && m.matchNode(state, n.Type) && m.matchNode(state, n.Body)
	case opFuncProtoDecl:
		n, ok := n.(*ast.FuncDecl)
		return ok && n.Recv == nil && n.Body == nil &&
			m.matchNode(state, n.Name) && m.matchNode(state, n.Type)
	case opMethodDecl:
		n, ok := n.(*ast.FuncDecl)
		return ok && n.Recv != nil && n.Body != nil &&
			m.matchNode(state, n.Recv) && m.matchNode(state, n.Name) && m.matchNode(state, n.Type) && m.matchNode(state, n.Body)
	case opMethodProtoDecl:
		n, ok := n.(*ast.FuncDecl)
		return ok && n.Recv != nil && n.Body == nil &&
			m.matchNode(state, n.Recv) && m.matchNode(state, n.Name) && m.matchNode(state, n.Type)

	case opValueSpec:
		n, ok := n.(*ast.ValueSpec)
		return ok && len(n.Values) == 0 && n.Type == nil &&
			len(n.Names) == 1 && m.matchNode(state, n.Names[0])
	case opValueInitSpec:
		n, ok := n.(*ast.ValueSpec)
		return ok && len(n.Values) != 0 && n.Type == nil &&
			m.matchIdentSlice(state, n.Names) && m.matchExprSlice(state, n.Values)
	case opTypedValueSpec:
		n, ok := n.(*ast.ValueSpec)
		return ok && len(n.Values) == 0 && n.Type != nil &&
			m.matchIdentSlice(state, n.Names) && m.matchNode(state, n.Type)
	case opTypedValueInitSpec:
		n, ok := n.(*ast.ValueSpec)
		return ok && len(n.Values) != 0 &&
			m.matchIdentSlice(state, n.Names) && m.matchNode(state, n.Type) && m.matchExprSlice(state, n.Values)

	case opTypeSpec:
		n, ok := n.(*ast.TypeSpec)
		return ok && !n.Assign.IsValid() && m.matchNode(state, n.Name) && m.matchNode(state, n.Type)
	case opTypeAliasSpec:
		n, ok := n.(*ast.TypeSpec)
		return ok && n.Assign.IsValid() && m.matchNode(state, n.Name) && m.matchNode(state, n.Type)

	case opDeclStmt:
		n, ok := n.(*ast.DeclStmt)
		return ok && m.matchNode(state, n.Decl)

	case opConstDecl:
		n, ok := n.(*ast.GenDecl)
		return ok && n.Tok == token.CONST && m.matchSpecSlice(state, n.Specs)
	case opVarDecl:
		n, ok := n.(*ast.GenDecl)
		return ok && n.Tok == token.VAR && m.matchSpecSlice(state, n.Specs)
	case opTypeDecl:
		n, ok := n.(*ast.GenDecl)
		return ok && n.Tok == token.TYPE && m.matchSpecSlice(state, n.Specs)
	case opAnyImportDecl:
		n, ok := n.(*ast.GenDecl)
		return ok && n.Tok == token.IMPORT
	case opImportDecl:
		n, ok := n.(*ast.GenDecl)
		return ok && n.Tok == token.IMPORT && m.matchSpecSlice(state, n.Specs)

	case opEmptyPackage:
		n, ok := n.(*ast.File)
		return ok && len(n.Imports) == 0 && len(n.Decls) == 0 && m.matchNode(state, n.Name)

	default:
		panic(fmt.Sprintf("unexpected op %s", inst.op))
	}
}

func (m *matcher) matchNode(state *MatcherState, n ast.Node) bool {
	return m.matchNodeWithInst(state, m.nextInst(state), n)
}

func (m *matcher) matchArgList(state *MatcherState, exprs []ast.Expr) bool {
	inst := m.nextInst(state)
	if inst.op != opSimpleArgList {
		return m.matchExprSlice(state, exprs)
	}
	if len(exprs) != int(inst.value) {
		return false
	}
	for _, x := range exprs {
		if !m.matchNode(state, x) {
			return false
		}
	}
	return true
}

func (m *matcher) matchStmtSlice(state *MatcherState, stmts []ast.Stmt) bool {
	matched, _ := m.matchNodeList(state, stmtSlice(stmts), false)
	return matched != nil
}

func (m *matcher) matchExprSlice(state *MatcherState, exprs []ast.Expr) bool {
	matched, _ := m.matchNodeList(state, ExprSlice(exprs), false)
	return matched != nil
}

func (m *matcher) matchFieldSlice(state *MatcherState, fields []*ast.Field) bool {
	matched, _ := m.matchNodeList(state, fieldSlice(fields), false)
	return matched != nil
}

func (m *matcher) matchIdentSlice(state *MatcherState, idents []*ast.Ident) bool {
	matched, _ := m.matchNodeList(state, identSlice(idents), false)
	return matched != nil
}

func (m *matcher) matchSpecSlice(state *MatcherState, specs []ast.Spec) bool {
	matched, _ := m.matchNodeList(state, specSlice(specs), false)
	return matched != nil
}

// matchNodeList matches two lists of nodes. It uses a common algorithm to match
// wildcard patterns with any number of nodes without recursion.
func (m *matcher) matchNodeList(state *MatcherState, nodes NodeSlice, partial bool) (matched ast.Node, offset int) {
	sliceLen := nodes.Len()
	inst := m.nextInst(state)
	if inst.op == opEnd {
		if sliceLen == 0 {
			return nodes, 0
		}
		return nil, -1
	}
	pcBase := state.pc
	pcNext := 0
	j := 0
	jNext := 0
	partialStart, partialEnd := 0, sliceLen

	type restart struct {
		matches   []CapturedNode
		pc        int
		j         int
		wildStart int
		wildName  string
	}
	// We need to stack these because otherwise some edge cases
	// would not match properly. Since we have various kinds of
	// wildcards (nodes containing them, $_, and $*_), in some cases
	// we may have to go back and do multiple restarts to get to the
	// right starting position.
	var stack []restart
	wildName := ""
	wildStart := 0
	push := func(next int) {
		if next > sliceLen {
			return // would be discarded anyway
		}
		pcNext = state.pc - 1
		jNext = next
		stack = append(stack, restart{state.capture, pcNext, next, wildStart, wildName})
	}
	pop := func() {
		j = jNext
		state.pc = pcNext
		state.capture = stack[len(stack)-1].matches
		wildName = stack[len(stack)-1].wildName
		wildStart = stack[len(stack)-1].wildStart
		stack = stack[:len(stack)-1]
		pcNext = 0
		jNext = 0
		if len(stack) != 0 {
			pcNext = stack[len(stack)-1].pc
			jNext = stack[len(stack)-1].j
		}
	}

	// wouldMatch returns whether the current wildcard - if any -
	// matches the nodes we are currently trying it on.
	wouldMatch := func() bool {
		switch wildName {
		case "", "_":
			return true
		}
		return m.matchNamed(state, wildName, nodes.slice(wildStart, j))
	}
	for ; inst.op != opEnd || j < sliceLen; inst = m.nextInst(state) {
		if inst.op != opEnd {
			if inst.op == opNodeSeq || inst.op == opNamedNodeSeq {
				// keep track of where this wildcard
				// started (if name == wildName,
				// we're trying the same wildcard
				// matching one more node)
				name := "_"
				if inst.op == opNamedNodeSeq {
					name = m.stringValue(inst)
				}
				if name != wildName {
					wildStart = j
					wildName = name
				}
				// try to match zero or more at j,
				// restarting at j+1 if it fails
				push(j + 1)
				continue
			}
			if partial && state.pc == pcBase {
				// let "b; c" match "a; b; c"
				// (simulates a $*_ at the beginning)
				partialStart = j
				push(j + 1)
			}
			if j < sliceLen && wouldMatch() && m.matchNodeWithInst(state, inst, nodes.At(j)) {
				// ordinary match
				wildName = ""
				j++
				continue
			}
		}
		if partial && inst.op == opEnd && wildName == "" {
			partialEnd = j
			break // let "b; c" match "b; c; d"
		}
		// mismatch, try to restart
		if 0 < jNext && jNext <= sliceLen && (state.pc != pcNext || j != jNext) {
			pop()
			continue
		}
		return nil, -1
	}
	if !wouldMatch() {
		return nil, -1
	}
	return nodes.slice(partialStart, partialEnd), partialEnd + 1
}

func findNamed(capture []CapturedNode, name string) (ast.Node, bool) {
	for _, c := range capture {
		if c.Name == name {
			return c.Node, true
		}
	}
	return nil, false
}

func literalValue(lit *ast.BasicLit) interface{} {
	switch lit.Kind {
	case token.INT:
		v, err := strconv.ParseInt(lit.Value, 0, 64)
		if err == nil {
			return v
		}
	case token.CHAR:
		s, err := strconv.Unquote(lit.Value)
		if err != nil {
			return nil
		}
		// Return the first rune.
		for _, c := range s {
			return c
		}
	case token.STRING:
		s, err := strconv.Unquote(lit.Value)
		if err == nil {
			return s
		}
	case token.FLOAT:
		v, err := strconv.ParseFloat(lit.Value, 64)
		if err == nil {
			return v
		}
	case token.IMAG:
		v, err := strconv.ParseComplex(lit.Value, 128)
		if err == nil {
			return v
		}
	}
	return nil
}

func equalNodes(x, y ast.Node) bool {
	if x == nil || y == nil {
		return x == y
	}
	switch x := x.(type) {
	case stmtSlice:
		y, ok := y.(stmtSlice)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !astequal.Stmt(x[i], y[i]) {
				return false
			}
		}
		return true
	case ExprSlice:
		y, ok := y.(ExprSlice)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !astequal.Expr(x[i], y[i]) {
				return false
			}
		}
		return true
	case declSlice:
		y, ok := y.(declSlice)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !astequal.Decl(x[i], y[i]) {
				return false
			}
		}
		return true

	default:
		return astequal.Node(x, y)
	}
}

func toStmtSlice(nodes ...ast.Node) stmtSlice {
	var stmts []ast.Stmt
	for _, node := range nodes {
		switch x := node.(type) {
		case nil:
		case ast.Stmt:
			stmts = append(stmts, x)
		case ast.Expr:
			stmts = append(stmts, &ast.ExprStmt{X: x})
		default:
			panic(fmt.Sprintf("unexpected node type: %T", x))
		}
	}
	return stmtSlice(stmts)
}
