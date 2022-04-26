package gogrep

import (
	"fmt"
	"go/ast"
	"go/token"

	"github.com/quasilyte/gogrep/internal/stdinfo"
)

type compileError string

func (e compileError) Error() string { return string(e) }

type compiler struct {
	config CompileConfig

	prog          *program
	stringIndexes map[string]uint8
	ifaceIndexes  map[interface{}]uint8

	info *PatternInfo

	insideStmtList bool
}

func (c *compiler) Compile(root ast.Node, info *PatternInfo) (p *program, err error) {
	defer func() {
		if err != nil {
			return
		}
		rv := recover()
		if rv == nil {
			return
		}
		if parseErr, ok := rv.(compileError); ok {
			err = parseErr
			return
		}
		panic(rv) // Not our panic
	}()

	c.info = info
	c.prog = &program{
		insts: make([]instruction, 0, 8),
	}
	c.stringIndexes = make(map[string]uint8)
	c.ifaceIndexes = make(map[interface{}]uint8)

	c.compileNode(root)

	if len(c.prog.insts) == 0 {
		return nil, c.errorf(root, "0 instructions generated")
	}

	return c.prog, nil
}

func (c *compiler) errorf(n ast.Node, format string, args ...interface{}) compileError {
	loc := c.config.Fset.Position(n.Pos())
	message := fmt.Sprintf("%s:%d: %s", loc.Filename, loc.Line, fmt.Sprintf(format, args...))
	return compileError(message)
}

func (c *compiler) toUint8(n ast.Node, v int) uint8 {
	if !fitsUint8(v) {
		panic(c.errorf(n, "implementation error: %v can't be converted to uint8", v))
	}
	return uint8(v)
}

func (c *compiler) internVar(n ast.Node, s string) uint8 {
	c.info.Vars[s] = struct{}{}
	index := c.internString(n, s)
	return index
}

func (c *compiler) internString(n ast.Node, s string) uint8 {
	if index, ok := c.stringIndexes[s]; ok {
		return index
	}
	index := len(c.prog.strings)
	if !fitsUint8(index) {
		panic(c.errorf(n, "implementation limitation: too many string values"))
	}
	c.stringIndexes[s] = uint8(index)
	c.prog.strings = append(c.prog.strings, s)
	return uint8(index)
}

func (c *compiler) internIface(n ast.Node, v interface{}) uint8 {
	if index, ok := c.ifaceIndexes[v]; ok {
		return index
	}
	index := len(c.prog.ifaces)
	if !fitsUint8(index) {
		panic(c.errorf(n, "implementation limitation: too many values"))
	}
	c.ifaceIndexes[v] = uint8(index)
	c.prog.ifaces = append(c.prog.ifaces, v)
	return uint8(index)
}

func (c *compiler) emitInst(inst instruction) {
	c.prog.insts = append(c.prog.insts, inst)
}

func (c *compiler) emitInstOp(op operation) {
	c.emitInst(instruction{op: op})
}

func (c *compiler) compileNode(n ast.Node) {
	switch n := n.(type) {
	case *ast.File:
		c.compileFile(n)
	case ast.Decl:
		c.compileDecl(n)
	case ast.Expr:
		c.compileExpr(n)
	case ast.Stmt:
		c.compileStmt(n)
	case *ast.ValueSpec:
		c.compileValueSpec(n)
	case stmtSlice:
		c.compileStmtSlice(n)
	case declSlice:
		c.compileDeclSlice(n)
	case ExprSlice:
		c.compileExprSlice(n)
	default:
		panic(c.errorf(n, "compileNode: unexpected %T", n))
	}
}

func (c *compiler) compileOptStmt(n ast.Stmt) {
	if exprStmt, ok := n.(*ast.ExprStmt); ok {
		if ident, ok := exprStmt.X.(*ast.Ident); ok && isWildName(ident.Name) {
			c.compileWildIdent(ident, true)
			return
		}
	}
	c.compileStmt(n)
}

func (c *compiler) compileOptExpr(n ast.Expr) {
	if ident, ok := n.(*ast.Ident); ok && isWildName(ident.Name) {
		c.compileWildIdent(ident, true)
		return
	}
	c.compileExpr(n)
}

func (c *compiler) compileOptFieldList(n *ast.FieldList) {
	if len(n.List) == 1 {
		if ident, ok := n.List[0].Type.(*ast.Ident); ok && isWildName(ident.Name) && len(n.List[0].Names) == 0 {
			// `func (...) $*result` - result could be anything
			// `func (...) $result`  - result is a field list of 1 element
			info := decodeWildName(ident.Name)
			switch {
			case info.Seq:
				c.compileWildIdent(ident, true)
			case info.Name == "_":
				c.emitInstOp(opFieldNode)
			default:
				c.emitInst(instruction{
					op:         opNamedFieldNode,
					valueIndex: c.internVar(n, info.Name),
				})
			}
			return
		}
	}
	c.compileFieldList(n)
}

func (c *compiler) compileFieldList(n *ast.FieldList) {
	c.emitInstOp(opFieldList)
	for _, x := range n.List {
		c.compileField(x)
	}
	c.emitInstOp(opEnd)
}

func (c *compiler) compileField(n *ast.Field) {
	switch {
	case len(n.Names) == 0:
		if ident, ok := n.Type.(*ast.Ident); ok && isWildName(ident.Name) {
			c.compileWildIdent(ident, false)
			return
		}
		c.emitInstOp(opUnnamedField)
	case len(n.Names) == 1:
		name := n.Names[0]
		if isWildName(name.Name) {
			c.emitInstOp(opField)
			c.compileWildIdent(name, false)
		} else {
			c.emitInst(instruction{
				op:         opSimpleField,
				valueIndex: c.internString(name, name.Name),
			})
		}
	default:
		c.emitInstOp(opMultiField)
		for _, name := range n.Names {
			c.compileIdent(name)
		}
		c.emitInstOp(opEnd)
	}
	c.compileExpr(n.Type)
}

func (c *compiler) compileValueSpec(spec *ast.ValueSpec) {
	switch {
	case spec.Type == nil && len(spec.Values) == 0:
		if isWildName(spec.Names[0].String()) {
			c.compileIdent(spec.Names[0])
			return
		}
		c.emitInstOp(opValueSpec)
	case spec.Type == nil:
		c.emitInstOp(opValueInitSpec)
	case len(spec.Values) == 0:
		c.emitInstOp(opTypedValueSpec)
	default:
		c.emitInstOp(opTypedValueInitSpec)
	}
	for _, name := range spec.Names {
		c.compileIdent(name)
	}
	c.emitInstOp(opEnd)
	if spec.Type != nil {
		c.compileOptExpr(spec.Type)
	}
	if len(spec.Values) != 0 {
		for _, v := range spec.Values {
			c.compileExpr(v)
		}
		c.emitInstOp(opEnd)
	}
}

func (c *compiler) compileTypeSpec(spec *ast.TypeSpec) {
	c.emitInstOp(pickOp(spec.Assign.IsValid(), opTypeAliasSpec, opTypeSpec))
	c.compileIdent(spec.Name)
	c.compileExpr(spec.Type)
}

func (c *compiler) compileFile(n *ast.File) {
	if len(n.Imports) == 0 && len(n.Decls) == 0 {
		c.emitInstOp(opEmptyPackage)
		c.compileIdent(n.Name)
		return
	}

	panic(c.errorf(n, "compileFile: unsupported file pattern"))
}

func (c *compiler) compileDecl(n ast.Decl) {
	switch n := n.(type) {
	case *ast.FuncDecl:
		c.compileFuncDecl(n)
	case *ast.GenDecl:
		c.compileGenDecl(n)

	default:
		panic(c.errorf(n, "compileDecl: unexpected %T", n))
	}
}

func (c *compiler) compileFuncDecl(n *ast.FuncDecl) {
	if n.Recv == nil {
		c.emitInstOp(pickOp(n.Body == nil, opFuncProtoDecl, opFuncDecl))
	} else {
		c.emitInstOp(pickOp(n.Body == nil, opMethodProtoDecl, opMethodDecl))
	}

	if n.Recv != nil {
		c.compileFieldList(n.Recv)
	}
	c.compileIdent(n.Name)
	c.compileFuncType(n.Type)
	if n.Body != nil {
		c.compileBlockStmt(n.Body)
	}
}

func (c *compiler) compileGenDecl(n *ast.GenDecl) {
	if c.insideStmtList {
		c.emitInstOp(opDeclStmt)
	}

	switch n.Tok {
	case token.CONST, token.VAR:
		c.emitInstOp(pickOp(n.Tok == token.CONST, opConstDecl, opVarDecl))
		for _, spec := range n.Specs {
			c.compileValueSpec(spec.(*ast.ValueSpec))
		}
		c.emitInstOp(opEnd)
	case token.TYPE:
		c.emitInstOp(opTypeDecl)
		for _, spec := range n.Specs {
			c.compileTypeSpec(spec.(*ast.TypeSpec))
		}
		c.emitInstOp(opEnd)

	default:
		panic(c.errorf(n, "unexpected gen decl"))
	}
}

func (c *compiler) compileExpr(n ast.Expr) {
	switch n := n.(type) {
	case *ast.BasicLit:
		c.compileBasicLit(n)
	case *ast.BinaryExpr:
		c.compileBinaryExpr(n)
	case *ast.IndexExpr:
		c.compileIndexExpr(n)
	case *ast.Ident:
		c.compileIdent(n)
	case *ast.CallExpr:
		c.compileCallExpr(n)
	case *ast.UnaryExpr:
		c.compileUnaryExpr(n)
	case *ast.StarExpr:
		c.compileStarExpr(n)
	case *ast.ParenExpr:
		c.compileParenExpr(n)
	case *ast.SliceExpr:
		c.compileSliceExpr(n)
	case *ast.StructType:
		c.compileStructType(n)
	case *ast.InterfaceType:
		c.compileInterfaceType(n)
	case *ast.FuncType:
		c.compileFuncType(n)
	case *ast.ArrayType:
		c.compileArrayType(n)
	case *ast.MapType:
		c.compileMapType(n)
	case *ast.ChanType:
		c.compileChanType(n)
	case *ast.CompositeLit:
		c.compileCompositeLit(n)
	case *ast.FuncLit:
		c.compileFuncLit(n)
	case *ast.Ellipsis:
		c.compileEllipsis(n)
	case *ast.KeyValueExpr:
		c.compileKeyValueExpr(n)
	case *ast.SelectorExpr:
		c.compileSelectorExpr(n)
	case *ast.TypeAssertExpr:
		c.compileTypeAssertExpr(n)

	default:
		panic(c.errorf(n, "compileExpr: unexpected %T", n))
	}
}

func (c *compiler) compileBasicLit(n *ast.BasicLit) {
	if !c.config.Strict {
		v := literalValue(n)
		if v == nil {
			panic(c.errorf(n, "can't convert %s (%s) value", n.Value, n.Kind))
		}
		c.prog.insts = append(c.prog.insts, instruction{
			op:         opBasicLit,
			valueIndex: c.internIface(n, v),
		})
		return
	}

	var inst instruction
	switch n.Kind {
	case token.INT:
		inst.op = opStrictIntLit
	case token.FLOAT:
		inst.op = opStrictFloatLit
	case token.STRING:
		inst.op = opStrictStringLit
	case token.CHAR:
		inst.op = opStrictCharLit
	default:
		inst.op = opStrictComplexLit
	}
	inst.valueIndex = c.internString(n, n.Value)
	c.prog.insts = append(c.prog.insts, inst)
}

func (c *compiler) compileBinaryExpr(n *ast.BinaryExpr) {
	c.prog.insts = append(c.prog.insts, instruction{
		op:    opBinaryExpr,
		value: c.toUint8(n, int(n.Op)),
	})
	c.compileExpr(n.X)
	c.compileExpr(n.Y)
}

func (c *compiler) compileIndexExpr(n *ast.IndexExpr) {
	c.emitInstOp(opIndexExpr)
	c.compileExpr(n.X)
	c.compileExpr(n.Index)
}

func (c *compiler) compileWildIdent(n *ast.Ident, optional bool) {
	info := decodeWildName(n.Name)
	var inst instruction
	switch {
	case info.Name == "_" && !info.Seq:
		inst.op = opNode
	case info.Name == "_" && info.Seq:
		inst.op = pickOp(optional, opOptNode, opNodeSeq)
	case info.Name != "_" && !info.Seq:
		inst.op = opNamedNode
		inst.valueIndex = c.internVar(n, info.Name)
	default:
		inst.op = pickOp(optional, opNamedOptNode, opNamedNodeSeq)
		inst.valueIndex = c.internVar(n, info.Name)
	}
	c.prog.insts = append(c.prog.insts, inst)
}

func (c *compiler) compileIdent(n *ast.Ident) {
	if isWildName(n.Name) {
		c.compileWildIdent(n, false)
		return
	}

	c.prog.insts = append(c.prog.insts, instruction{
		op:         opIdent,
		valueIndex: c.internString(n, n.Name),
	})
}

func (c *compiler) compileExprMembers(list []ast.Expr) {
	isSimple := len(list) <= 255
	if isSimple {
		for _, x := range list {
			if decodeWildNode(x).Seq {
				isSimple = false
				break
			}
		}
	}

	if isSimple {
		c.emitInst(instruction{
			op:    opSimpleArgList,
			value: uint8(len(list)),
		})
		for _, x := range list {
			c.compileExpr(x)
		}
	} else {
		c.emitInstOp(opArgList)
		for _, x := range list {
			c.compileExpr(x)
		}
		c.emitInstOp(opEnd)
	}
}

func (c *compiler) compileCallExpr(n *ast.CallExpr) {
	canBeVariadic := func(n *ast.CallExpr) bool {
		if len(n.Args) == 0 {
			return false
		}
		lastArg, ok := n.Args[len(n.Args)-1].(*ast.Ident)
		if !ok {
			return false
		}
		return isWildName(lastArg.Name) && decodeWildName(lastArg.Name).Seq
	}

	op := opNonVariadicCallExpr
	if n.Ellipsis.IsValid() {
		op = opVariadicCallExpr
	} else if canBeVariadic(n) {
		op = opCallExpr
	}

	c.emitInstOp(op)
	c.compileSymbol(n.Fun)
	c.compileExprMembers(n.Args)
}

// compileSymbol is mostly like a normal compileExpr, but it's used
// in places where we can find a type/function symbol.
//
// For example, in function call expressions a called function expression
// can look like `fmt.Sprint`. It will be compiled as a special
// selector expression that requires `fmt` to be a package as opposed
// to only check that it's an identifier with "fmt" value.
func (c *compiler) compileSymbol(sym ast.Expr) {
	compilePkgSymbol := func(c *compiler, sym ast.Expr) bool {
		e, ok := sym.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		ident, ok := e.X.(*ast.Ident)
		if !ok || isWildName(e.Sel.Name) {
			return false
		}
		pkgPath := c.config.Imports[ident.Name]
		if pkgPath == "" && stdinfo.Packages[ident.Name] != "" {
			pkgPath = stdinfo.Packages[ident.Name]
		}
		if pkgPath == "" {
			return false
		}
		c.emitInst(instruction{
			op:         opSimpleSelectorExpr,
			valueIndex: c.internString(e.Sel, e.Sel.String()),
		})
		c.emitInst(instruction{
			op:         opPkg,
			valueIndex: c.internString(ident, pkgPath),
		})
		return true
	}

	if c.config.WithTypes {
		if compilePkgSymbol(c, sym) {
			return
		}
	}

	c.compileExpr(sym)
}

func (c *compiler) compileUnaryExpr(n *ast.UnaryExpr) {
	c.prog.insts = append(c.prog.insts, instruction{
		op:    opUnaryExpr,
		value: c.toUint8(n, int(n.Op)),
	})
	c.compileExpr(n.X)
}

func (c *compiler) compileStarExpr(n *ast.StarExpr) {
	c.emitInstOp(opStarExpr)
	c.compileExpr(n.X)
}

func (c *compiler) compileParenExpr(n *ast.ParenExpr) {
	c.emitInstOp(opParenExpr)
	c.compileExpr(n.X)
}

func (c *compiler) compileSliceExpr(n *ast.SliceExpr) {
	switch {
	case n.Low == nil && n.High == nil && !n.Slice3:
		c.emitInstOp(opSliceExpr)
		c.compileOptExpr(n.X)
	case n.Low != nil && n.High == nil && !n.Slice3:
		c.emitInstOp(opSliceFromExpr)
		c.compileOptExpr(n.X)
		c.compileOptExpr(n.Low)
	case n.Low == nil && n.High != nil && !n.Slice3:
		c.emitInstOp(opSliceToExpr)
		c.compileOptExpr(n.X)
		c.compileOptExpr(n.High)
	case n.Low != nil && n.High != nil && !n.Slice3:
		c.emitInstOp(opSliceFromToExpr)
		c.compileOptExpr(n.X)
		c.compileOptExpr(n.Low)
		c.compileOptExpr(n.High)
	case n.Low == nil && n.Slice3:
		c.emitInstOp(opSliceToCapExpr)
		c.compileOptExpr(n.X)
		c.compileOptExpr(n.High)
		c.compileOptExpr(n.Max)
	case n.Low != nil && n.Slice3:
		c.emitInstOp(opSliceFromToCapExpr)
		c.compileOptExpr(n.X)
		c.compileOptExpr(n.Low)
		c.compileOptExpr(n.High)
		c.compileOptExpr(n.Max)
	default:
		panic(c.errorf(n, "unexpected slice expr"))
	}
}

func (c *compiler) compileStructType(n *ast.StructType) {
	c.emitInstOp(opStructType)
	c.compileOptFieldList(n.Fields)
}

func (c *compiler) compileInterfaceType(n *ast.InterfaceType) {
	c.emitInstOp(opInterfaceType)
	c.compileOptFieldList(n.Methods)
}

func (c *compiler) compileFuncType(n *ast.FuncType) {
	void := n.Results == nil || len(n.Results.List) == 0
	if void {
		c.emitInstOp(opVoidFuncType)
	} else {
		c.emitInstOp(opFuncType)
	}
	c.compileOptFieldList(n.Params)
	if !void {
		c.compileOptFieldList(n.Results)
	}
}

func (c *compiler) compileArrayType(n *ast.ArrayType) {
	if n.Len == nil {
		c.emitInstOp(opSliceType)
		c.compileExpr(n.Elt)
	} else {
		c.emitInstOp(opArrayType)
		c.compileExpr(n.Len)
		c.compileExpr(n.Elt)
	}
}

func (c *compiler) compileMapType(n *ast.MapType) {
	c.emitInstOp(opMapType)
	c.compileExpr(n.Key)
	c.compileExpr(n.Value)
}

func (c *compiler) compileChanType(n *ast.ChanType) {
	c.emitInst(instruction{
		op:    opChanType,
		value: c.toUint8(n, int(n.Dir)),
	})
	c.compileExpr(n.Value)
}

func (c *compiler) compileCompositeLit(n *ast.CompositeLit) {
	if n.Type == nil {
		c.emitInstOp(opCompositeLit)
	} else {
		c.emitInstOp(opTypedCompositeLit)
		c.compileExpr(n.Type)
	}
	for _, elt := range n.Elts {
		c.compileExpr(elt)
	}
	c.emitInstOp(opEnd)
}

func (c *compiler) compileFuncLit(n *ast.FuncLit) {
	c.emitInstOp(opFuncLit)
	c.compileFuncType(n.Type)
	c.compileBlockStmt(n.Body)
}

func (c *compiler) compileEllipsis(n *ast.Ellipsis) {
	if n.Elt == nil {
		c.emitInstOp(opEllipsis)
	} else {
		c.emitInstOp(opTypedEllipsis)
		c.compileExpr(n.Elt)
	}
}

func (c *compiler) compileKeyValueExpr(n *ast.KeyValueExpr) {
	c.emitInstOp(opKeyValueExpr)
	c.compileExpr(n.Key)
	c.compileExpr(n.Value)
}

func (c *compiler) compileSelectorExpr(n *ast.SelectorExpr) {
	if isWildName(n.Sel.Name) {
		c.emitInstOp(opSelectorExpr)
		c.compileWildIdent(n.Sel, false)
		c.compileExpr(n.X)
		return
	}

	c.prog.insts = append(c.prog.insts, instruction{
		op:         opSimpleSelectorExpr,
		valueIndex: c.internString(n.Sel, n.Sel.String()),
	})
	c.compileExpr(n.X)
}

func (c *compiler) compileTypeAssertExpr(n *ast.TypeAssertExpr) {
	if n.Type != nil {
		c.emitInstOp(opTypeAssertExpr)
		c.compileExpr(n.X)
		c.compileExpr(n.Type)
	} else {
		c.emitInstOp(opTypeSwitchAssertExpr)
		c.compileExpr(n.X)
	}
}

func (c *compiler) compileStmt(n ast.Stmt) {
	switch n := n.(type) {
	case *ast.AssignStmt:
		c.compileAssignStmt(n)
	case *ast.BlockStmt:
		c.compileBlockStmt(n)
	case *ast.ExprStmt:
		c.compileExprStmt(n)
	case *ast.IfStmt:
		c.compileIfStmt(n)
	case *ast.CaseClause:
		c.compileCaseClause(n)
	case *ast.SwitchStmt:
		c.compileSwitchStmt(n)
	case *ast.TypeSwitchStmt:
		c.compileTypeSwitchStmt(n)
	case *ast.SelectStmt:
		c.compileSelectStmt(n)
	case *ast.ForStmt:
		c.compileForStmt(n)
	case *ast.RangeStmt:
		c.compileRangeStmt(n)
	case *ast.IncDecStmt:
		c.compileIncDecStmt(n)
	case *ast.EmptyStmt:
		c.compileEmptyStmt(n)
	case *ast.ReturnStmt:
		c.compileReturnStmt(n)
	case *ast.BranchStmt:
		c.compileBranchStmt(n)
	case *ast.LabeledStmt:
		c.compileLabeledStmt(n)
	case *ast.GoStmt:
		c.compileGoStmt(n)
	case *ast.DeferStmt:
		c.compileDeferStmt(n)
	case *ast.SendStmt:
		c.compileSendStmt(n)
	case *ast.DeclStmt:
		c.compileDecl(n.Decl)

	default:
		panic(c.errorf(n, "compileStmt: unexpected %T", n))
	}
}

func (c *compiler) compileAssignStmt(n *ast.AssignStmt) {
	if len(n.Lhs) == 1 && len(n.Rhs) == 1 {
		lhsInfo := decodeWildNode(n.Lhs[0])
		rhsInfo := decodeWildNode(n.Rhs[0])
		if !lhsInfo.Seq && !rhsInfo.Seq {
			c.emitInst(instruction{
				op:    opAssignStmt,
				value: uint8(n.Tok),
			})
			c.compileExpr(n.Lhs[0])
			c.compileExpr(n.Rhs[0])
			return
		}
	}

	c.emitInst(instruction{
		op:    opMultiAssignStmt,
		value: uint8(n.Tok),
	})
	for _, x := range n.Lhs {
		c.compileExpr(x)
	}
	c.emitInstOp(opEnd)
	for _, x := range n.Rhs {
		c.compileExpr(x)
	}
	c.emitInstOp(opEnd)
}

func (c *compiler) compileBlockStmt(n *ast.BlockStmt) {
	c.emitInstOp(opBlockStmt)
	insideStmtList := c.insideStmtList
	c.insideStmtList = true
	for _, elt := range n.List {
		c.compileStmt(elt)
	}
	c.insideStmtList = insideStmtList
	c.emitInstOp(opEnd)
}

func (c *compiler) compileExprStmt(n *ast.ExprStmt) {
	if ident, ok := n.X.(*ast.Ident); ok && isWildName(ident.Name) {
		c.compileIdent(ident)
	} else {
		c.emitInstOp(opExprStmt)
		c.compileExpr(n.X)
	}
}

func (c *compiler) compileIfStmt(n *ast.IfStmt) {
	// Check for the special case: `if $*_ ...` should match all if statements.
	if ident, ok := n.Cond.(*ast.Ident); ok && n.Init == nil && isWildName(ident.Name) {
		info := decodeWildName(ident.Name)
		if info.Seq && info.Name == "_" {
			// Set Init to Cond, change cond from $*_ to $_.
			n.Init = &ast.ExprStmt{X: n.Cond}
			cond := &ast.Ident{Name: encodeWildName(info.Name, false)}
			n.Cond = cond
			c.compileIfStmt(n)
			return
		}
		// Named $* is harder and slower.
		if info.Seq {
			c.prog.insts = append(c.prog.insts, instruction{
				op:         pickOp(n.Else == nil, opIfNamedOptStmt, opIfNamedOptElseStmt),
				valueIndex: c.internVar(ident, info.Name),
			})
			c.compileStmt(n.Body)
			if n.Else != nil {
				c.compileStmt(n.Else)
			}
			return
		}
	}

	switch {
	case n.Init == nil && n.Else == nil:
		c.emitInstOp(opIfStmt)
		c.compileExpr(n.Cond)
		c.compileStmt(n.Body)
	case n.Init != nil && n.Else == nil:
		c.emitInstOp(opIfInitStmt)
		c.compileOptStmt(n.Init)
		c.compileExpr(n.Cond)
		c.compileStmt(n.Body)
	case n.Init == nil && n.Else != nil:
		c.emitInstOp(opIfElseStmt)
		c.compileExpr(n.Cond)
		c.compileStmt(n.Body)
		c.compileStmt(n.Else)
	case n.Init != nil && n.Else != nil:
		c.emitInstOp(opIfInitElseStmt)
		c.compileOptStmt(n.Init)
		c.compileExpr(n.Cond)
		c.compileStmt(n.Body)
		c.compileStmt(n.Else)

	default:
		panic(c.errorf(n, "unexpected if stmt"))
	}
}

func (c *compiler) compileCommClause(n *ast.CommClause) {
	c.emitInstOp(pickOp(n.Comm == nil, opDefaultCommClause, opCommClause))
	if n.Comm != nil {
		c.compileStmt(n.Comm)
	}
	for _, x := range n.Body {
		c.compileStmt(x)
	}
	c.emitInstOp(opEnd)
}

func (c *compiler) compileCaseClause(n *ast.CaseClause) {
	c.emitInstOp(pickOp(n.List == nil, opDefaultCaseClause, opCaseClause))
	if n.List != nil {
		for _, x := range n.List {
			c.compileExpr(x)
		}
		c.emitInstOp(opEnd)
	}
	for _, x := range n.Body {
		c.compileStmt(x)
	}
	c.emitInstOp(opEnd)
}

func (c *compiler) compileSwitchBody(n *ast.BlockStmt) {
	wildcardCase := func(cc *ast.CaseClause) *ast.Ident {
		if len(cc.List) != 1 || len(cc.Body) != 1 {
			return nil
		}
		v, ok := cc.List[0].(*ast.Ident)
		if !ok || !isWildName(v.Name) {
			return nil
		}
		bodyStmt, ok := cc.Body[0].(*ast.ExprStmt)
		if !ok {
			return nil
		}
		bodyIdent, ok := bodyStmt.X.(*ast.Ident)
		if !ok || bodyIdent.Name != "gogrep_body" {
			return nil
		}
		return v
	}
	for _, cc := range n.List {
		cc := cc.(*ast.CaseClause)
		wildcard := wildcardCase(cc)
		if wildcard == nil {
			c.compileCaseClause(cc)
			continue
		}
		c.compileWildIdent(wildcard, false)
	}
	c.emitInstOp(opEnd)
}

func (c *compiler) compileSwitchStmt(n *ast.SwitchStmt) {
	var op operation
	switch {
	case n.Init == nil && n.Tag == nil:
		op = opSwitchStmt
	case n.Init == nil && n.Tag != nil:
		op = opSwitchTagStmt
	case n.Init != nil && n.Tag == nil:
		op = opSwitchInitStmt
	default:
		op = opSwitchInitTagStmt
	}

	c.emitInstOp(op)
	if n.Init != nil {
		c.compileOptStmt(n.Init)
	}
	if n.Tag != nil {
		c.compileOptExpr(n.Tag)
	}
	c.compileSwitchBody(n.Body)
}

func (c *compiler) compileTypeSwitchStmt(n *ast.TypeSwitchStmt) {
	c.emitInstOp(pickOp(n.Init == nil, opTypeSwitchStmt, opTypeSwitchInitStmt))
	if n.Init != nil {
		c.compileOptStmt(n.Init)
	}
	c.compileStmt(n.Assign)
	c.compileSwitchBody(n.Body)
}

func (c *compiler) compileSelectStmt(n *ast.SelectStmt) {
	c.emitInstOp(opSelectStmt)

	wildcardCase := func(cc *ast.CommClause) *ast.Ident {
		if cc.Comm == nil {
			return nil
		}
		vStmt, ok := cc.Comm.(*ast.ExprStmt)
		if !ok {
			return nil
		}
		v, ok := vStmt.X.(*ast.Ident)
		if !ok || !isWildName(v.Name) {
			return nil
		}
		bodyStmt, ok := cc.Body[0].(*ast.ExprStmt)
		if !ok {
			return nil
		}
		bodyIdent, ok := bodyStmt.X.(*ast.Ident)
		if !ok || bodyIdent.Name != "gogrep_body" {
			return nil
		}
		return v
	}
	for _, cc := range n.Body.List {
		cc := cc.(*ast.CommClause)
		wildcard := wildcardCase(cc)
		if wildcard == nil {
			c.compileCommClause(cc)
			continue
		}
		c.compileWildIdent(wildcard, false)
	}
	c.emitInstOp(opEnd)
}

func (c *compiler) compileForStmt(n *ast.ForStmt) {
	var op operation
	switch {
	case n.Init == nil && n.Cond == nil && n.Post == nil:
		op = opForStmt
	case n.Init == nil && n.Cond == nil && n.Post != nil:
		op = opForPostStmt
	case n.Init == nil && n.Cond != nil && n.Post == nil:
		op = opForCondStmt
	case n.Init == nil && n.Cond != nil && n.Post != nil:
		op = opForCondPostStmt
	case n.Init != nil && n.Cond == nil && n.Post == nil:
		op = opForInitStmt
	case n.Init != nil && n.Cond == nil && n.Post != nil:
		op = opForInitPostStmt
	case n.Init != nil && n.Cond != nil && n.Post == nil:
		op = opForInitCondStmt
	default:
		op = opForInitCondPostStmt
	}

	c.emitInstOp(op)
	if n.Init != nil {
		c.compileOptStmt(n.Init)
	}
	if n.Cond != nil {
		c.compileOptExpr(n.Cond)
	}
	if n.Post != nil {
		c.compileOptStmt(n.Post)
	}
	c.compileBlockStmt(n.Body)
}

func (c *compiler) compileRangeStmt(n *ast.RangeStmt) {
	switch {
	case n.Key == nil && n.Value == nil:
		c.emitInstOp(opRangeStmt)
		c.compileExpr(n.X)
		c.compileStmt(n.Body)
	case n.Key != nil && n.Value == nil:
		c.emitInst(instruction{
			op:    opRangeKeyStmt,
			value: c.toUint8(n, int(n.Tok)),
		})
		c.compileExpr(n.Key)
		c.compileExpr(n.X)
		c.compileStmt(n.Body)
	case n.Key != nil && n.Value != nil:
		c.emitInst(instruction{
			op:    opRangeKeyValueStmt,
			value: c.toUint8(n, int(n.Tok)),
		})
		c.compileExpr(n.Key)
		c.compileExpr(n.Value)
		c.compileExpr(n.X)
		c.compileStmt(n.Body)
	default:
		panic(c.errorf(n, "unexpected range stmt"))
	}
}

func (c *compiler) compileIncDecStmt(n *ast.IncDecStmt) {
	c.prog.insts = append(c.prog.insts, instruction{
		op:    opIncDecStmt,
		value: c.toUint8(n, int(n.Tok)),
	})
	c.compileExpr(n.X)
}

func (c *compiler) compileEmptyStmt(n *ast.EmptyStmt) {
	_ = n // unused
	c.emitInstOp(opEmptyStmt)
}

func (c *compiler) compileReturnStmt(n *ast.ReturnStmt) {
	c.emitInstOp(opReturnStmt)
	for _, x := range n.Results {
		c.compileExpr(x)
	}
	c.emitInstOp(opEnd)
}

func (c *compiler) compileBranchStmt(n *ast.BranchStmt) {
	if n.Label != nil {
		if isWildName(n.Label.Name) {
			c.prog.insts = append(c.prog.insts, instruction{
				op:    opLabeledBranchStmt,
				value: c.toUint8(n, int(n.Tok)),
			})
			c.compileWildIdent(n.Label, false)
		} else {
			c.prog.insts = append(c.prog.insts, instruction{
				op:         opSimpleLabeledBranchStmt,
				value:      c.toUint8(n, int(n.Tok)),
				valueIndex: c.internString(n.Label, n.Label.Name),
			})
		}
		return
	}
	c.prog.insts = append(c.prog.insts, instruction{
		op:    opBranchStmt,
		value: c.toUint8(n, int(n.Tok)),
	})
}

func (c *compiler) compileLabeledStmt(n *ast.LabeledStmt) {
	if isWildName(n.Label.Name) {
		c.emitInstOp(opLabeledStmt)
		c.compileWildIdent(n.Label, false)
		c.compileStmt(n.Stmt)
		return
	}

	c.prog.insts = append(c.prog.insts, instruction{
		op:         opSimpleLabeledStmt,
		valueIndex: c.internString(n.Label, n.Label.Name),
	})
	c.compileStmt(n.Stmt)
}

func (c *compiler) compileGoStmt(n *ast.GoStmt) {
	c.emitInstOp(opGoStmt)
	c.compileExpr(n.Call)
}

func (c *compiler) compileDeferStmt(n *ast.DeferStmt) {
	c.emitInstOp(opDeferStmt)
	c.compileExpr(n.Call)
}

func (c *compiler) compileSendStmt(n *ast.SendStmt) {
	c.emitInstOp(opSendStmt)
	c.compileExpr(n.Chan)
	c.compileExpr(n.Value)
}

func (c *compiler) compileDeclSlice(decls declSlice) {
	c.emitInstOp(opMultiDecl)
	for _, n := range decls {
		c.compileDecl(n)
	}
	c.emitInstOp(opEnd)
}

func (c *compiler) compileStmtSlice(stmts stmtSlice) {
	c.emitInstOp(opMultiStmt)
	insideStmtList := c.insideStmtList
	c.insideStmtList = true
	for _, n := range stmts {
		c.compileStmt(n)
	}
	c.insideStmtList = insideStmtList
	c.emitInstOp(opEnd)
}

func (c *compiler) compileExprSlice(exprs ExprSlice) {
	c.emitInstOp(opMultiExpr)
	for _, n := range exprs {
		c.compileExpr(n)
	}
	c.emitInstOp(opEnd)
}

func pickOp(cond bool, ifTrue, ifFalse operation) operation {
	if cond {
		return ifTrue
	}
	return ifFalse
}

func fitsUint8(v int) bool {
	return v >= 0 && v <= 0xff
}
