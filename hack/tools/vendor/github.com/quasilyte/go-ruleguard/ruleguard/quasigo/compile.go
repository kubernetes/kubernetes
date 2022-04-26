package quasigo

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"

	"github.com/quasilyte/go-ruleguard/ruleguard/goutil"
	"golang.org/x/tools/go/ast/astutil"
)

var voidType = &types.Tuple{}

func compile(ctx *CompileContext, fn *ast.FuncDecl) (compiled *Func, err error) {
	defer func() {
		if err != nil {
			return
		}
		rv := recover()
		if rv == nil {
			return
		}
		if compileErr, ok := rv.(compileError); ok {
			err = compileErr
			return
		}
		panic(rv) // not our panic
	}()

	return compileFunc(ctx, fn), nil
}

func compileFunc(ctx *CompileContext, fn *ast.FuncDecl) *Func {
	cl := compiler{
		ctx:              ctx,
		fnType:           ctx.Types.ObjectOf(fn.Name).Type().(*types.Signature),
		constantsPool:    make(map[interface{}]int),
		intConstantsPool: make(map[int]int),
		locals:           make(map[string]int),
	}
	return cl.compileFunc(fn)
}

type compiler struct {
	ctx *CompileContext

	fnType  *types.Signature
	retType types.Type

	lastOp opcode

	locals           map[string]int
	constantsPool    map[interface{}]int
	intConstantsPool map[int]int
	params           map[string]int

	code         []byte
	constants    []interface{}
	intConstants []int

	breakTarget    *label
	continueTarget *label

	labels []*label
}

type label struct {
	targetPos int
	sources   []int
}

type compileError string

func (e compileError) Error() string { return string(e) }

func (cl *compiler) compileFunc(fn *ast.FuncDecl) *Func {
	switch cl.fnType.Results().Len() {
	case 0:
		cl.retType = voidType
	case 1:
		cl.retType = cl.fnType.Results().At(0).Type()
	default:
		panic(cl.errorf(fn.Name, "multi-result functions are not supported"))
	}

	if !cl.isSupportedType(cl.retType) {
		panic(cl.errorUnsupportedType(fn.Name, cl.retType, "function result"))
	}

	dbg := funcDebugInfo{
		paramNames: make([]string, cl.fnType.Params().Len()),
	}

	cl.params = make(map[string]int, cl.fnType.Params().Len())
	for i := 0; i < cl.fnType.Params().Len(); i++ {
		p := cl.fnType.Params().At(i)
		paramName := p.Name()
		paramType := p.Type()
		cl.params[paramName] = i
		dbg.paramNames[i] = paramName
		if !cl.isSupportedType(paramType) {
			panic(cl.errorUnsupportedType(fn.Name, paramType, paramName+" param"))
		}
	}

	cl.compileStmt(fn.Body)
	if cl.retType == voidType {
		cl.emit(opReturn)
	}

	compiled := &Func{
		code:         cl.code,
		constants:    cl.constants,
		intConstants: cl.intConstants,
	}
	if len(cl.locals) != 0 {
		dbg.localNames = make([]string, len(cl.locals))
		for localName, localIndex := range cl.locals {
			dbg.localNames[localIndex] = localName
		}
	}
	cl.ctx.Env.debug.funcs[compiled] = dbg
	cl.linkJumps()
	return compiled
}

func (cl *compiler) compileStmt(stmt ast.Stmt) {
	switch stmt := stmt.(type) {
	case *ast.ReturnStmt:
		cl.compileReturnStmt(stmt)

	case *ast.AssignStmt:
		cl.compileAssignStmt(stmt)

	case *ast.IncDecStmt:
		cl.compileIncDecStmt(stmt)

	case *ast.IfStmt:
		cl.compileIfStmt(stmt)

	case *ast.ForStmt:
		cl.compileForStmt(stmt)

	case *ast.BranchStmt:
		cl.compileBranchStmt(stmt)

	case *ast.ExprStmt:
		cl.compileExprStmt(stmt)

	case *ast.BlockStmt:
		for i := range stmt.List {
			cl.compileStmt(stmt.List[i])
		}

	default:
		panic(cl.errorf(stmt, "can't compile %T yet", stmt))
	}
}

func (cl *compiler) compileIncDecStmt(stmt *ast.IncDecStmt) {
	varname, ok := stmt.X.(*ast.Ident)
	if !ok {
		panic(cl.errorf(stmt.X, "can assign only to simple variables"))
	}
	id := cl.getLocal(varname, varname.String())
	if stmt.Tok == token.INC {
		cl.emit8(opIncLocal, id)
	} else {
		cl.emit8(opDecLocal, id)
	}
}

func (cl *compiler) compileBranchStmt(branch *ast.BranchStmt) {
	if branch.Label != nil {
		panic(cl.errorf(branch.Label, "can't compile %s with a label", branch.Tok))
	}

	switch branch.Tok {
	case token.BREAK:
		cl.emitJump(opJump, cl.breakTarget)
	default:
		panic(cl.errorf(branch, "can't compile %s yet", branch.Tok))
	}
}

func (cl *compiler) compileExprStmt(stmt *ast.ExprStmt) {
	if call, ok := stmt.X.(*ast.CallExpr); ok {
		sig := cl.ctx.Types.TypeOf(call.Fun).(*types.Signature)
		if sig.Results() != nil {
			panic(cl.errorf(call, "only void funcs can be used in stmt context"))
		}
		cl.compileCallExpr(call)
		return
	}

	panic(cl.errorf(stmt.X, "can't compile this expr stmt yet: %T", stmt.X))
}

func (cl *compiler) compileForStmt(stmt *ast.ForStmt) {
	labelBreak := cl.newLabel()
	labelContinue := cl.newLabel()
	prevBreakTarget := cl.breakTarget
	prevContinueTarget := cl.continueTarget
	cl.breakTarget = labelBreak
	cl.continueTarget = labelContinue

	switch {
	case stmt.Cond != nil && stmt.Init != nil && stmt.Post != nil:
		// Will be implemented later; probably when the max number of locals will be lifted.
		panic(cl.errorf(stmt, "can't compile C-style for loops yet"))

	case stmt.Cond != nil && stmt.Init == nil && stmt.Post == nil:
		// `for <cond> { ... }`
		labelBody := cl.newLabel()
		cl.emitJump(opJump, labelContinue)
		cl.bindLabel(labelBody)
		cl.compileStmt(stmt.Body)
		cl.bindLabel(labelContinue)
		cl.compileExpr(stmt.Cond)
		cl.emitJump(opJumpTrue, labelBody)
		cl.bindLabel(labelBreak)

	default:
		// `for { ... }`
		cl.bindLabel(labelContinue)
		cl.compileStmt(stmt.Body)
		cl.emitJump(opJump, labelContinue)
		cl.bindLabel(labelBreak)
	}

	cl.breakTarget = prevBreakTarget
	cl.continueTarget = prevContinueTarget
}

func (cl *compiler) compileIfStmt(stmt *ast.IfStmt) {
	if stmt.Else == nil {
		labelEnd := cl.newLabel()
		cl.compileExpr(stmt.Cond)
		cl.emitJump(opJumpFalse, labelEnd)
		cl.compileStmt(stmt.Body)
		cl.bindLabel(labelEnd)
		return
	}

	labelEnd := cl.newLabel()
	labelElse := cl.newLabel()
	cl.compileExpr(stmt.Cond)
	cl.emitJump(opJumpFalse, labelElse)
	cl.compileStmt(stmt.Body)
	if !cl.isUncondJump(cl.lastOp) {
		cl.emitJump(opJump, labelEnd)
	}
	cl.bindLabel(labelElse)
	cl.compileStmt(stmt.Else)
	cl.bindLabel(labelEnd)
}

func (cl *compiler) compileAssignStmt(assign *ast.AssignStmt) {
	if len(assign.Rhs) != 1 {
		panic(cl.errorf(assign, "only single right operand is allowed in assignments"))
	}
	for _, lhs := range assign.Lhs {
		_, ok := lhs.(*ast.Ident)
		if !ok {
			panic(cl.errorf(lhs, "can assign only to simple variables"))
		}
	}

	rhs := assign.Rhs[0]
	cl.compileExpr(rhs)

	if assign.Tok == token.DEFINE {
		for i := len(assign.Lhs) - 1; i >= 0; i-- {
			varname := assign.Lhs[i].(*ast.Ident)
			typ := cl.ctx.Types.TypeOf(varname)
			if _, ok := cl.locals[varname.String()]; ok {
				panic(cl.errorf(varname, "%s variable shadowing is not allowed", varname))
			}
			if !cl.isSupportedType(typ) {
				panic(cl.errorUnsupportedType(varname, typ, varname.String()+" local variable"))
			}
			if len(cl.locals) == maxFuncLocals {
				panic(cl.errorf(varname, "can't define %s: too many locals", varname))
			}
			id := len(cl.locals)
			cl.locals[varname.String()] = id
			cl.emit8(pickOp(typeIsInt(typ), opSetIntLocal, opSetLocal), id)
		}
	} else {
		for i := len(assign.Lhs) - 1; i >= 0; i-- {
			varname := assign.Lhs[i].(*ast.Ident)
			typ := cl.ctx.Types.TypeOf(varname)
			id := cl.getLocal(varname, varname.String())
			cl.emit8(pickOp(typeIsInt(typ), opSetIntLocal, opSetLocal), id)
		}
	}
}

func (cl *compiler) getLocal(v ast.Expr, varname string) int {
	id, ok := cl.locals[varname]
	if !ok {
		if _, ok := cl.params[varname]; ok {
			panic(cl.errorf(v, "can't assign to %s, params are readonly", varname))
		}
		panic(cl.errorf(v, "%s is not a writeable local variable", varname))
	}
	return id
}

func (cl *compiler) compileReturnStmt(ret *ast.ReturnStmt) {
	if cl.retType == voidType {
		cl.emit(opReturn)
		return
	}

	if ret.Results == nil {
		panic(cl.errorf(ret, "'naked' return statements are not allowed"))
	}

	switch {
	case identName(ret.Results[0]) == "true":
		cl.emit(opReturnTrue)
	case identName(ret.Results[0]) == "false":
		cl.emit(opReturnFalse)
	default:
		cl.compileExpr(ret.Results[0])
		typ := cl.ctx.Types.TypeOf(ret.Results[0])
		cl.emit(pickOp(typeIsInt(typ), opReturnIntTop, opReturnTop))
	}
}

func (cl *compiler) compileExpr(e ast.Expr) {
	cv := cl.ctx.Types.Types[e].Value
	if cv != nil {
		cl.compileConstantValue(e, cv)
		return
	}

	switch e := e.(type) {
	case *ast.ParenExpr:
		cl.compileExpr(e.X)

	case *ast.Ident:
		cl.compileIdent(e)

	case *ast.SelectorExpr:
		cl.compileSelectorExpr(e)

	case *ast.UnaryExpr:
		switch e.Op {
		case token.NOT:
			cl.compileUnaryOp(opNot, e)
		default:
			panic(cl.errorf(e, "can't compile unary %s yet", e.Op))
		}

	case *ast.SliceExpr:
		cl.compileSliceExpr(e)

	case *ast.BinaryExpr:
		cl.compileBinaryExpr(e)

	case *ast.CallExpr:
		cl.compileCallExpr(e)

	default:
		panic(cl.errorf(e, "can't compile %T yet", e))
	}
}

func (cl *compiler) compileSelectorExpr(e *ast.SelectorExpr) {
	typ := cl.ctx.Types.TypeOf(e.X)
	key := funcKey{
		name:      e.Sel.String(),
		qualifier: typ.String(),
	}

	if funcID, ok := cl.ctx.Env.nameToNativeFuncID[key]; ok {
		cl.compileExpr(e.X)
		cl.emit16(opCallNative, int(funcID))
		return
	}

	panic(cl.errorf(e, "can't compile %s field access", e.Sel))
}

func (cl *compiler) compileBinaryExpr(e *ast.BinaryExpr) {
	typ := cl.ctx.Types.TypeOf(e.X)

	switch e.Op {
	case token.LOR:
		cl.compileOr(e)
	case token.LAND:
		cl.compileAnd(e)

	case token.NEQ:
		switch {
		case identName(e.X) == "nil":
			cl.compileExpr(e.Y)
			cl.emit(opIsNotNil)
		case identName(e.Y) == "nil":
			cl.compileExpr(e.X)
			cl.emit(opIsNotNil)
		case typeIsString(typ):
			cl.compileBinaryOp(opNotEqString, e)
		case typeIsInt(typ):
			cl.compileBinaryOp(opNotEqInt, e)
		default:
			panic(cl.errorf(e, "!= is not implemented for %s operands", typ))
		}
	case token.EQL:
		switch {
		case identName(e.X) == "nil":
			cl.compileExpr(e.Y)
			cl.emit(opIsNil)
		case identName(e.Y) == "nil":
			cl.compileExpr(e.X)
			cl.emit(opIsNil)
		case typeIsString(cl.ctx.Types.TypeOf(e.X)):
			cl.compileBinaryOp(opEqString, e)
		case typeIsInt(cl.ctx.Types.TypeOf(e.X)):
			cl.compileBinaryOp(opEqInt, e)
		default:
			panic(cl.errorf(e, "== is not implemented for %s operands", typ))
		}

	case token.GTR:
		cl.compileIntBinaryOp(e, opGtInt, typ)
	case token.GEQ:
		cl.compileIntBinaryOp(e, opGtEqInt, typ)
	case token.LSS:
		cl.compileIntBinaryOp(e, opLtInt, typ)
	case token.LEQ:
		cl.compileIntBinaryOp(e, opLtEqInt, typ)

	case token.ADD:
		switch {
		case typeIsString(typ):
			cl.compileBinaryOp(opConcat, e)
		case typeIsInt(typ):
			cl.compileBinaryOp(opAdd, e)
		default:
			panic(cl.errorf(e, "+ is not implemented for %s operands", typ))
		}

	case token.SUB:
		cl.compileIntBinaryOp(e, opSub, typ)

	default:
		panic(cl.errorf(e, "can't compile binary %s yet", e.Op))
	}
}

func (cl *compiler) compileIntBinaryOp(e *ast.BinaryExpr, op opcode, typ types.Type) {
	switch {
	case typeIsInt(typ):
		cl.compileBinaryOp(op, e)
	default:
		panic(cl.errorf(e, "%s is not implemented for %s operands", e.Op, typ))
	}
}

func (cl *compiler) compileSliceExpr(slice *ast.SliceExpr) {
	if slice.Slice3 {
		panic(cl.errorf(slice, "can't compile 3-index slicing"))
	}

	// No need to do slicing, its no-op `s[:]`.
	if slice.Low == nil && slice.High == nil {
		cl.compileExpr(slice.X)
		return
	}

	sliceOp := opStringSlice
	sliceFromOp := opStringSliceFrom
	sliceToOp := opStringSliceTo

	if !typeIsString(cl.ctx.Types.TypeOf(slice.X)) {
		panic(cl.errorf(slice.X, "can't compile slicing of something that is not a string"))
	}

	switch {
	case slice.Low == nil && slice.High != nil:
		cl.compileExpr(slice.X)
		cl.compileExpr(slice.High)
		cl.emit(sliceToOp)
	case slice.Low != nil && slice.High == nil:
		cl.compileExpr(slice.X)
		cl.compileExpr(slice.Low)
		cl.emit(sliceFromOp)
	default:
		cl.compileExpr(slice.X)
		cl.compileExpr(slice.Low)
		cl.compileExpr(slice.High)
		cl.emit(sliceOp)
	}
}

func (cl *compiler) compileBuiltinCall(fn *ast.Ident, call *ast.CallExpr) {
	switch fn.Name {
	case `len`:
		s := call.Args[0]
		cl.compileExpr(s)
		if !typeIsString(cl.ctx.Types.TypeOf(s)) {
			panic(cl.errorf(s, "can't compile len() with non-string argument yet"))
		}
		cl.emit(opStringLen)

	case `println`:
		if len(call.Args) != 1 {
			panic(cl.errorf(call, "only 1-arg form of println() is supported"))
		}
		funcName := "Print"
		if typeIsInt(cl.ctx.Types.TypeOf(call.Args[0])) {
			funcName = "PrintInt"
		}
		key := funcKey{qualifier: "builtin", name: funcName}
		if !cl.compileNativeCall(key, 0, nil, call.Args) {
			panic(cl.errorf(fn, "builtin.%s native func is not registered", funcName))
		}

	default:
		panic(cl.errorf(fn, "can't compile %s() builtin function call yet", fn))
	}
}

func (cl *compiler) compileCallExpr(call *ast.CallExpr) {
	if id, ok := astutil.Unparen(call.Fun).(*ast.Ident); ok {
		_, isBuiltin := cl.ctx.Types.ObjectOf(id).(*types.Builtin)
		if isBuiltin {
			cl.compileBuiltinCall(id, call)
			return
		}
	}

	expr, fn := goutil.ResolveFunc(cl.ctx.Types, call.Fun)
	if fn == nil {
		panic(cl.errorf(call.Fun, "can't resolve the called function"))
	}

	// TODO: just use Func.FullName as a key?
	key := funcKey{name: fn.Name()}
	sig := fn.Type().(*types.Signature)
	if sig.Recv() != nil {
		key.qualifier = sig.Recv().Type().String()
	} else {
		key.qualifier = fn.Pkg().Path()
	}
	variadic := 0
	if sig.Variadic() {
		variadic = sig.Params().Len() - 1
	}
	if !cl.compileNativeCall(key, variadic, expr, call.Args) {
		panic(cl.errorf(call.Fun, "can't compile a call to %s func", key))
	}
}

func (cl *compiler) compileNativeCall(key funcKey, variadic int, expr ast.Expr, args []ast.Expr) bool {
	funcID, ok := cl.ctx.Env.nameToNativeFuncID[key]
	if !ok {
		return false
	}
	if expr != nil {
		cl.compileExpr(expr)
	}
	if len(args) == 1 {
		// Check that it's not a f(g()) call, where g() returns
		// a multi-value result; we can't compile that yet.
		if call, ok := args[0].(*ast.CallExpr); ok {
			results := cl.ctx.Types.TypeOf(call.Fun).(*types.Signature).Results()
			if results != nil && results.Len() > 1 {
				panic(cl.errorf(args[0], "can't pass tuple as a func argument"))
			}
		}
	}

	normalArgs := args
	var variadicArgs []ast.Expr
	if variadic != 0 {
		normalArgs = args[:variadic]
		variadicArgs = args[variadic:]
	}

	for _, arg := range normalArgs {
		cl.compileExpr(arg)
	}
	if variadic != 0 {
		for _, arg := range variadicArgs {
			cl.compileExpr(arg)
			// int-typed values should appear in the interface{}-typed
			// objects slice, so we get all variadic args placed in one place.
			if typeIsInt(cl.ctx.Types.TypeOf(arg)) {
				cl.emit(opConvIntToIface)
			}
		}
		if len(variadicArgs) > 255 {
			panic(cl.errorf(expr, "too many variadic args"))
		}
		// Even if len(variadicArgs) is 0, we still need to overwrite
		// the old variadicLen value, so the variadic func is not confused
		// by some unrelated value.
		cl.emit8(opSetVariadicLen, len(variadicArgs))
	}

	cl.emit16(opCallNative, int(funcID))
	return true
}

func (cl *compiler) compileUnaryOp(op opcode, e *ast.UnaryExpr) {
	cl.compileExpr(e.X)
	cl.emit(op)
}

func (cl *compiler) compileBinaryOp(op opcode, e *ast.BinaryExpr) {
	cl.compileExpr(e.X)
	cl.compileExpr(e.Y)
	cl.emit(op)
}

func (cl *compiler) compileOr(e *ast.BinaryExpr) {
	labelEnd := cl.newLabel()
	cl.compileExpr(e.X)
	cl.emit(opDup)
	cl.emitJump(opJumpTrue, labelEnd)
	cl.compileExpr(e.Y)
	cl.bindLabel(labelEnd)
}

func (cl *compiler) compileAnd(e *ast.BinaryExpr) {
	labelEnd := cl.newLabel()
	cl.compileExpr(e.X)
	cl.emit(opDup)
	cl.emitJump(opJumpFalse, labelEnd)
	cl.compileExpr(e.Y)
	cl.bindLabel(labelEnd)
}

func (cl *compiler) compileIdent(ident *ast.Ident) {
	tv := cl.ctx.Types.Types[ident]
	cv := tv.Value
	if cv != nil {
		cl.compileConstantValue(ident, cv)
		return
	}
	if paramIndex, ok := cl.params[ident.String()]; ok {
		cl.emit8(pickOp(typeIsInt(tv.Type), opPushIntParam, opPushParam), paramIndex)
		return
	}
	if localIndex, ok := cl.locals[ident.String()]; ok {
		cl.emit8(pickOp(typeIsInt(tv.Type), opPushIntLocal, opPushLocal), localIndex)
		return
	}

	panic(cl.errorf(ident, "can't compile a %s (type %s) variable read", ident.String(), tv.Type))
}

func (cl *compiler) compileConstantValue(source ast.Expr, cv constant.Value) {
	switch cv.Kind() {
	case constant.Bool:
		v := constant.BoolVal(cv)
		if v {
			cl.emit(opPushTrue)
		} else {
			cl.emit(opPushFalse)
		}

	case constant.String:
		v := constant.StringVal(cv)
		id := cl.internConstant(v)
		cl.emit8(opPushConst, id)

	case constant.Int:
		v, exact := constant.Int64Val(cv)
		if !exact {
			panic(cl.errorf(source, "non-exact int value"))
		}
		id := cl.internIntConstant(int(v))
		cl.emit8(opPushIntConst, id)

	case constant.Complex:
		panic(cl.errorf(source, "can't compile complex number constants yet"))

	case constant.Float:
		panic(cl.errorf(source, "can't compile float constants yet"))

	default:
		panic(cl.errorf(source, "unexpected constant %v", cv))
	}
}

func (cl *compiler) internIntConstant(v int) int {
	if id, ok := cl.intConstantsPool[v]; ok {
		return id
	}
	id := len(cl.intConstants)
	cl.intConstants = append(cl.intConstants, v)
	cl.intConstantsPool[v] = id
	return id
}

func (cl *compiler) internConstant(v interface{}) int {
	if _, ok := v.(int); ok {
		panic("compiler error: int constant interned as interface{}")
	}
	if id, ok := cl.constantsPool[v]; ok {
		return id
	}
	id := len(cl.constants)
	cl.constants = append(cl.constants, v)
	cl.constantsPool[v] = id
	return id
}

func (cl *compiler) linkJumps() {
	for _, l := range cl.labels {
		for _, jumpPos := range l.sources {
			offset := l.targetPos - jumpPos
			patchPos := jumpPos + 1
			put16(cl.code, patchPos, offset)
		}
	}
}

func (cl *compiler) newLabel() *label {
	l := &label{}
	cl.labels = append(cl.labels, l)
	return l
}

func (cl *compiler) bindLabel(l *label) {
	l.targetPos = len(cl.code)
}

func (cl *compiler) emit(op opcode) {
	cl.lastOp = op
	cl.code = append(cl.code, byte(op))
}

func (cl *compiler) emitJump(op opcode, l *label) {
	l.sources = append(l.sources, len(cl.code))
	cl.emit(op)
	cl.code = append(cl.code, 0, 0)
}

func (cl *compiler) emit8(op opcode, arg8 int) {
	cl.emit(op)
	cl.code = append(cl.code, byte(arg8))
}

func (cl *compiler) emit16(op opcode, arg16 int) {
	cl.emit(op)
	buf := make([]byte, 2)
	put16(buf, 0, arg16)
	cl.code = append(cl.code, buf...)
}

func (cl *compiler) errorUnsupportedType(e ast.Node, typ types.Type, where string) compileError {
	return cl.errorf(e, "%s type: %s is not supported, try something simpler", where, typ)
}

func (cl *compiler) errorf(n ast.Node, format string, args ...interface{}) compileError {
	loc := cl.ctx.Fset.Position(n.Pos())
	message := fmt.Sprintf("%s:%d: %s", loc.Filename, loc.Line, fmt.Sprintf(format, args...))
	return compileError(message)
}

func (cl *compiler) isUncondJump(op opcode) bool {
	switch op {
	case opJump, opReturnFalse, opReturnTrue, opReturnTop, opReturnIntTop:
		return true
	default:
		return false
	}
}

func (cl *compiler) isSupportedType(typ types.Type) bool {
	if typ == voidType {
		return true
	}

	switch typ := typ.Underlying().(type) {
	case *types.Pointer:
		// 1. Pointers to structs are supported.
		_, isStruct := typ.Elem().Underlying().(*types.Struct)
		return isStruct

	case *types.Basic:
		// 2. Some of the basic types are supported.
		// TODO: support byte/uint8 and maybe float64.
		switch typ.Kind() {
		case types.Bool, types.Int, types.String:
			return true
		default:
			return false
		}

	case *types.Interface:
		// 3. Interfaces are supported.
		return true

	default:
		return false
	}
}
