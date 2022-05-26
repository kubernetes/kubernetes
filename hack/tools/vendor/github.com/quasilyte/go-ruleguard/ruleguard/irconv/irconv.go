package irconv

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"path"
	"strconv"
	"strings"

	"github.com/go-toolsmith/astcopy"
	"github.com/quasilyte/go-ruleguard/ruleguard/goutil"
	"github.com/quasilyte/go-ruleguard/ruleguard/ir"
	"golang.org/x/tools/go/ast/astutil"
)

type Context struct {
	Pkg   *types.Package
	Types *types.Info
	Fset  *token.FileSet
	Src   []byte
}

func ConvertFile(ctx *Context, f *ast.File) (result *ir.File, err error) {
	defer func() {
		if err != nil {
			return
		}
		rv := recover()
		if rv == nil {
			return
		}
		if convErr, ok := rv.(convError); ok {
			err = convErr.err
			return
		}
		panic(rv) // not our panic
	}()

	conv := &converter{
		types: ctx.Types,
		pkg:   ctx.Pkg,
		fset:  ctx.Fset,
		src:   ctx.Src,
	}
	result = conv.ConvertFile(f)
	return result, nil
}

type convError struct {
	err error
}

type localMacroFunc struct {
	name     string
	params   []string
	template ast.Expr
}

type converter struct {
	types *types.Info
	pkg   *types.Package
	fset  *token.FileSet
	src   []byte

	group      *ir.RuleGroup
	groupFuncs []localMacroFunc

	dslPkgname string // The local name of the "ruleguard/dsl" package (usually its just "dsl")
}

func (conv *converter) errorf(n ast.Node, format string, args ...interface{}) convError {
	loc := conv.fset.Position(n.Pos())
	msg := fmt.Sprintf(format, args...)
	return convError{err: fmt.Errorf("%s:%d: %s", loc.Filename, loc.Line, msg)}
}

func (conv *converter) ConvertFile(f *ast.File) *ir.File {
	result := &ir.File{
		PkgPath: conv.pkg.Path(),
	}

	conv.dslPkgname = "dsl"

	for _, imp := range f.Imports {
		importPath, err := strconv.Unquote(imp.Path.Value)
		if err != nil {
			panic(conv.errorf(imp, "unquote %s import path: %s", imp.Path.Value, err))
		}
		if importPath == "github.com/quasilyte/go-ruleguard/dsl" {
			if imp.Name != nil {
				conv.dslPkgname = imp.Name.Name
			}
		}
	}

	for _, decl := range f.Decls {
		funcDecl, ok := decl.(*ast.FuncDecl)
		if !ok {
			genDecl := decl.(*ast.GenDecl)
			if genDecl.Tok != token.IMPORT {
				conv.addCustomDecl(result, decl)
			}
			continue
		}

		if funcDecl.Name.String() == "init" {
			conv.convertInitFunc(result, funcDecl)
			continue
		}

		if conv.isMatcherFunc(funcDecl) {
			result.RuleGroups = append(result.RuleGroups, *conv.convertRuleGroup(funcDecl))
		} else {
			conv.addCustomDecl(result, funcDecl)
		}
	}

	return result
}

func (conv *converter) convertInitFunc(dst *ir.File, decl *ast.FuncDecl) {
	for _, stmt := range decl.Body.List {
		exprStmt, ok := stmt.(*ast.ExprStmt)
		if !ok {
			panic(conv.errorf(stmt, "unsupported statement"))
		}
		call, ok := exprStmt.X.(*ast.CallExpr)
		if !ok {
			panic(conv.errorf(stmt, "unsupported expr"))
		}
		fn, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			panic(conv.errorf(stmt, "unsupported call"))
		}
		pkg, ok := fn.X.(*ast.Ident)
		if !ok || pkg.Name != conv.dslPkgname {
			panic(conv.errorf(stmt, "unsupported call"))
		}

		switch fn.Sel.Name {
		case "ImportRules":
			prefix := conv.parseStringArg(call.Args[0])
			bundleSelector, ok := call.Args[1].(*ast.SelectorExpr)
			if !ok {
				panic(conv.errorf(call.Args[1], "expected a `pkgname.Bundle` argument"))
			}
			bundleObj := conv.types.ObjectOf(bundleSelector.Sel)
			dst.BundleImports = append(dst.BundleImports, ir.BundleImport{
				Prefix:  prefix,
				PkgPath: bundleObj.Pkg().Path(),
				Line:    conv.fset.Position(exprStmt.Pos()).Line,
			})

		default:
			panic(conv.errorf(stmt, "unsupported %s call", fn.Sel.Name))
		}
	}
}

func (conv *converter) addCustomDecl(dst *ir.File, decl ast.Decl) {
	begin := conv.fset.Position(decl.Pos())
	end := conv.fset.Position(decl.End())
	src := conv.src[begin.Offset:end.Offset]
	dst.CustomDecls = append(dst.CustomDecls, string(src))
}

func (conv *converter) isMatcherFunc(f *ast.FuncDecl) bool {
	typ := conv.types.ObjectOf(f.Name).Type().(*types.Signature)
	return typ.Results().Len() == 0 &&
		typ.Params().Len() == 1 &&
		typ.Params().At(0).Type().String() == "github.com/quasilyte/go-ruleguard/dsl.Matcher"
}

func (conv *converter) convertRuleGroup(decl *ast.FuncDecl) *ir.RuleGroup {
	result := &ir.RuleGroup{
		Line: conv.fset.Position(decl.Name.Pos()).Line,
	}
	conv.group = result
	conv.groupFuncs = conv.groupFuncs[:0]

	result.Name = decl.Name.String()
	result.MatcherName = decl.Type.Params.List[0].Names[0].String()

	if decl.Doc != nil {
		conv.convertDocComments(decl.Doc)
	}

	seenRules := false
	for _, stmt := range decl.Body.List {
		if assign, ok := stmt.(*ast.AssignStmt); ok && assign.Tok == token.DEFINE {
			conv.localDefine(assign)
			continue
		}

		if _, ok := stmt.(*ast.DeclStmt); ok {
			continue
		}
		stmtExpr, ok := stmt.(*ast.ExprStmt)
		if !ok {
			panic(conv.errorf(stmt, "expected a %s method call, found %s", result.MatcherName, goutil.SprintNode(conv.fset, stmt)))
		}
		call, ok := stmtExpr.X.(*ast.CallExpr)
		if !ok {
			panic(conv.errorf(stmt, "expected a %s method call, found %s", result.MatcherName, goutil.SprintNode(conv.fset, stmt)))
		}

		switch conv.matcherMethodName(call) {
		case "Import":
			if seenRules {
				panic(conv.errorf(call, "Import() should be used before any rules definitions"))
			}
			conv.doMatcherImport(call)
		default:
			seenRules = true
			conv.convertRuleExpr(call)
		}
	}

	return result
}

func (conv *converter) findLocalMacro(call *ast.CallExpr) *localMacroFunc {
	fn, ok := call.Fun.(*ast.Ident)
	if !ok {
		return nil
	}
	for i := range conv.groupFuncs {
		if conv.groupFuncs[i].name == fn.Name {
			return &conv.groupFuncs[i]
		}
	}
	return nil
}

func (conv *converter) expandMacro(macro *localMacroFunc, call *ast.CallExpr) ir.FilterExpr {
	// Check that call args are OK.
	// Since "function calls" are implemented as a macro expansion here,
	// we don't allow arguments that have a non-trivial evaluation.
	isSafe := func(arg ast.Expr) bool {
		switch arg := astutil.Unparen(arg).(type) {
		case *ast.BasicLit, *ast.Ident:
			return true

		case *ast.IndexExpr:
			mapIdent, ok := astutil.Unparen(arg.X).(*ast.Ident)
			if !ok {
				return false
			}
			if mapIdent.Name != conv.group.MatcherName {
				return false
			}
			key, ok := astutil.Unparen(arg.Index).(*ast.BasicLit)
			if !ok || key.Kind != token.STRING {
				return false
			}
			return true

		default:
			return false
		}
	}
	args := map[string]ast.Expr{}
	for i, arg := range call.Args {
		paramName := macro.params[i]
		if !isSafe(arg) {
			panic(conv.errorf(arg, "unsupported/too complex %s argument", paramName))
		}
		args[paramName] = astutil.Unparen(arg)
	}

	body := astcopy.Expr(macro.template)
	expanded := astutil.Apply(body, nil, func(cur *astutil.Cursor) bool {
		if ident, ok := cur.Node().(*ast.Ident); ok {
			arg, ok := args[ident.Name]
			if ok {
				cur.Replace(arg)
				return true
			}
		}
		// astcopy above will copy the AST tree, but it won't update
		// the associated types.Info map of const values.
		// We'll try to solve that issue at least partially here.
		if lit, ok := cur.Node().(*ast.BasicLit); ok {
			switch lit.Kind {
			case token.STRING:
				val, err := strconv.Unquote(lit.Value)
				if err == nil {
					conv.types.Types[lit] = types.TypeAndValue{
						Type:  types.Typ[types.UntypedString],
						Value: constant.MakeString(val),
					}
				}
			case token.INT:
				val, err := strconv.ParseInt(lit.Value, 0, 64)
				if err == nil {
					conv.types.Types[lit] = types.TypeAndValue{
						Type:  types.Typ[types.UntypedInt],
						Value: constant.MakeInt64(val),
					}
				}
			case token.FLOAT:
				val, err := strconv.ParseFloat(lit.Value, 64)
				if err == nil {
					conv.types.Types[lit] = types.TypeAndValue{
						Type:  types.Typ[types.UntypedFloat],
						Value: constant.MakeFloat64(val),
					}
				}
			}
		}
		return true
	})

	return conv.convertFilterExpr(expanded.(ast.Expr))
}

func (conv *converter) localDefine(assign *ast.AssignStmt) {
	if len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
		panic(conv.errorf(assign, "multi-value := is not supported"))
	}
	lhs, ok := assign.Lhs[0].(*ast.Ident)
	if !ok {
		panic(conv.errorf(assign.Lhs[0], "only simple ident lhs is supported"))
	}
	rhs := assign.Rhs[0]
	fn, ok := rhs.(*ast.FuncLit)
	if !ok {
		panic(conv.errorf(rhs, "only func literals are supported on the rhs"))
	}
	typ := conv.types.TypeOf(fn).(*types.Signature)
	isBoolResult := typ.Results() != nil &&
		typ.Results().Len() == 1 &&
		typ.Results().At(0).Type() == types.Typ[types.Bool]
	if !isBoolResult {
		var loc ast.Node = fn.Type
		if fn.Type.Results != nil {
			loc = fn.Type.Results
		}
		panic(conv.errorf(loc, "only funcs returning bool are supported"))
	}
	if len(fn.Body.List) != 1 {
		panic(conv.errorf(fn.Body, "only simple 1 return statement funcs are supported"))
	}
	stmt, ok := fn.Body.List[0].(*ast.ReturnStmt)
	if !ok {
		panic(conv.errorf(fn.Body.List[0], "expected a return statement, found %T", fn.Body.List[0]))
	}
	var params []string
	for _, field := range fn.Type.Params.List {
		for _, id := range field.Names {
			params = append(params, id.Name)
		}
	}
	macro := localMacroFunc{
		name:     lhs.Name,
		params:   params,
		template: stmt.Results[0],
	}
	conv.groupFuncs = append(conv.groupFuncs, macro)
}

func (conv *converter) doMatcherImport(call *ast.CallExpr) {
	pkgPath := conv.parseStringArg(call.Args[0])
	pkgName := path.Base(pkgPath)
	conv.group.Imports = append(conv.group.Imports, ir.PackageImport{
		Path: pkgPath,
		Name: pkgName,
	})
}

func (conv *converter) matcherMethodName(call *ast.CallExpr) string {
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	id, ok := selector.X.(*ast.Ident)
	if !ok || id.Name != conv.group.MatcherName {
		return ""
	}
	return selector.Sel.Name
}

func (conv *converter) convertDocComments(comment *ast.CommentGroup) {
	knownPragmas := []string{
		"tags",
		"summary",
		"before",
		"after",
		"note",
	}

	for _, c := range comment.List {
		if !strings.HasPrefix(c.Text, "//doc:") {
			continue
		}
		s := strings.TrimPrefix(c.Text, "//doc:")
		var pragma string
		for i := range knownPragmas {
			if strings.HasPrefix(s, knownPragmas[i]) {
				pragma = knownPragmas[i]
				break
			}
		}
		if pragma == "" {
			panic(conv.errorf(c, "unrecognized 'doc' pragma in comment"))
		}
		s = strings.TrimPrefix(s, pragma)
		s = strings.TrimSpace(s)
		switch pragma {
		case "summary":
			conv.group.DocSummary = s
		case "before":
			conv.group.DocBefore = s
		case "after":
			conv.group.DocAfter = s
		case "note":
			conv.group.DocNote = s
		case "tags":
			conv.group.DocTags = strings.Fields(s)
		default:
			panic("unhandled 'doc' pragma: " + pragma) // Should never happen
		}
	}
}

func (conv *converter) convertRuleExpr(call *ast.CallExpr) {
	origCall := call
	var (
		matchArgs        *[]ast.Expr
		matchCommentArgs *[]ast.Expr
		whereArgs        *[]ast.Expr
		suggestArgs      *[]ast.Expr
		reportArgs       *[]ast.Expr
		atArgs           *[]ast.Expr
	)

	for {
		chain, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			break
		}
		switch chain.Sel.Name {
		case "Match":
			if matchArgs != nil {
				panic(conv.errorf(chain.Sel, "Match() can't be repeated"))
			}
			if matchCommentArgs != nil {
				panic(conv.errorf(chain.Sel, "Match() and MatchComment() can't be combined"))
			}
			matchArgs = &call.Args
		case "MatchComment":
			if matchCommentArgs != nil {
				panic(conv.errorf(chain.Sel, "MatchComment() can't be repeated"))
			}
			if matchArgs != nil {
				panic(conv.errorf(chain.Sel, "Match() and MatchComment() can't be combined"))
			}
			matchCommentArgs = &call.Args
		case "Where":
			if whereArgs != nil {
				panic(conv.errorf(chain.Sel, "Where() can't be repeated"))
			}
			whereArgs = &call.Args
		case "Suggest":
			if suggestArgs != nil {
				panic(conv.errorf(chain.Sel, "Suggest() can't be repeated"))
			}
			suggestArgs = &call.Args
		case "Report":
			if reportArgs != nil {
				panic(conv.errorf(chain.Sel, "Report() can't be repeated"))
			}
			reportArgs = &call.Args
		case "At":
			if atArgs != nil {
				panic(conv.errorf(chain.Sel, "At() can't be repeated"))
			}
			atArgs = &call.Args
		default:
			panic(conv.errorf(chain.Sel, "unexpected %s method", chain.Sel.Name))
		}
		call, ok = chain.X.(*ast.CallExpr)
		if !ok {
			break
		}
	}

	// AST patterns for Match() or regexp patterns for MatchComment().
	var alternatives []string
	var alternativeLines []int

	if matchArgs == nil && matchCommentArgs == nil {
		panic(conv.errorf(origCall, "missing Match() or MatchComment() call"))
	}

	if matchArgs != nil {
		for _, arg := range *matchArgs {
			alternatives = append(alternatives, conv.parseStringArg(arg))
			alternativeLines = append(alternativeLines, conv.fset.Position(arg.Pos()).Line)
		}
	} else {
		for _, arg := range *matchCommentArgs {
			alternatives = append(alternatives, conv.parseStringArg(arg))
			alternativeLines = append(alternativeLines, conv.fset.Position(arg.Pos()).Line)
		}
	}

	rule := ir.Rule{Line: conv.fset.Position(origCall.Pos()).Line}

	if atArgs != nil {
		index, ok := (*atArgs)[0].(*ast.IndexExpr)
		if !ok {
			panic(conv.errorf((*atArgs)[0], "expected %s[`varname`] expression", conv.group.MatcherName))
		}
		rule.LocationVar = conv.parseStringArg(index.Index)
	}

	if whereArgs != nil {
		rule.WhereExpr = conv.convertFilterExpr((*whereArgs)[0])
	}

	if suggestArgs != nil {
		rule.SuggestTemplate = conv.parseStringArg((*suggestArgs)[0])
	}

	if suggestArgs == nil && reportArgs == nil {
		panic(conv.errorf(origCall, "missing Report() or Suggest() call"))
	}
	if reportArgs == nil {
		rule.ReportTemplate = "suggestion: " + rule.SuggestTemplate
	} else {
		rule.ReportTemplate = conv.parseStringArg((*reportArgs)[0])
	}

	for i, alt := range alternatives {
		pat := ir.PatternString{
			Line:  alternativeLines[i],
			Value: alt,
		}
		if matchArgs != nil {
			rule.SyntaxPatterns = append(rule.SyntaxPatterns, pat)
		} else {
			rule.CommentPatterns = append(rule.CommentPatterns, pat)
		}
	}
	conv.group.Rules = append(conv.group.Rules, rule)
}

func (conv *converter) convertFilterExpr(e ast.Expr) ir.FilterExpr {
	result := conv.convertFilterExprImpl(e)
	result.Src = goutil.SprintNode(conv.fset, e)
	result.Line = conv.fset.Position(e.Pos()).Line
	if !result.IsValid() {
		panic(conv.errorf(e, "unsupported expr: %s (%T)", result.Src, e))
	}
	return result
}

func (conv *converter) convertFilterExprImpl(e ast.Expr) ir.FilterExpr {
	if cv := conv.types.Types[e].Value; cv != nil {
		switch cv.Kind() {
		case constant.String:
			v := constant.StringVal(cv)
			return ir.FilterExpr{Op: ir.FilterStringOp, Value: v}
		case constant.Int:
			v, ok := constant.Int64Val(cv)
			if ok {
				return ir.FilterExpr{Op: ir.FilterIntOp, Value: v}
			}
		}
	}
	convertExprList := func(list []ast.Expr) []ir.FilterExpr {
		if len(list) == 0 {
			return nil
		}
		result := make([]ir.FilterExpr, len(list))
		for i, e := range list {
			result[i] = conv.convertFilterExpr(e)
		}
		return result
	}

	switch e := e.(type) {
	case *ast.ParenExpr:
		return conv.convertFilterExpr(e.X)

	case *ast.UnaryExpr:
		x := conv.convertFilterExpr(e.X)
		args := []ir.FilterExpr{x}
		switch e.Op {
		case token.NOT:
			return ir.FilterExpr{Op: ir.FilterNotOp, Args: args}
		}

	case *ast.BinaryExpr:
		x := conv.convertFilterExpr(e.X)
		y := conv.convertFilterExpr(e.Y)
		args := []ir.FilterExpr{x, y}
		switch e.Op {
		case token.LAND:
			return ir.FilterExpr{Op: ir.FilterAndOp, Args: args}
		case token.LOR:
			return ir.FilterExpr{Op: ir.FilterOrOp, Args: args}
		case token.NEQ:
			return ir.FilterExpr{Op: ir.FilterNeqOp, Args: args}
		case token.EQL:
			return ir.FilterExpr{Op: ir.FilterEqOp, Args: args}
		case token.GTR:
			return ir.FilterExpr{Op: ir.FilterGtOp, Args: args}
		case token.LSS:
			return ir.FilterExpr{Op: ir.FilterLtOp, Args: args}
		case token.GEQ:
			return ir.FilterExpr{Op: ir.FilterGtEqOp, Args: args}
		case token.LEQ:
			return ir.FilterExpr{Op: ir.FilterLtEqOp, Args: args}
		default:
			panic(conv.errorf(e, "unexpected binary op: %s", e.Op.String()))
		}

	case *ast.SelectorExpr:
		op := conv.inspectFilterSelector(e)
		switch op.path {
		case "Text":
			return ir.FilterExpr{Op: ir.FilterVarTextOp, Value: op.varName}
		case "Line":
			return ir.FilterExpr{Op: ir.FilterVarLineOp, Value: op.varName}
		case "Pure":
			return ir.FilterExpr{Op: ir.FilterVarPureOp, Value: op.varName}
		case "Const":
			return ir.FilterExpr{Op: ir.FilterVarConstOp, Value: op.varName}
		case "ConstSlice":
			return ir.FilterExpr{Op: ir.FilterVarConstSliceOp, Value: op.varName}
		case "Addressable":
			return ir.FilterExpr{Op: ir.FilterVarAddressableOp, Value: op.varName}
		case "Type.Size":
			return ir.FilterExpr{Op: ir.FilterVarTypeSizeOp, Value: op.varName}
		}

	case *ast.CallExpr:
		op := conv.inspectFilterSelector(e)
		switch op.path {
		case "Deadcode":
			return ir.FilterExpr{Op: ir.FilterDeadcodeOp}
		case "GoVersion.Eq":
			return ir.FilterExpr{Op: ir.FilterGoVersionEqOp, Value: conv.parseStringArg(e.Args[0])}
		case "GoVersion.LessThan":
			return ir.FilterExpr{Op: ir.FilterGoVersionLessThanOp, Value: conv.parseStringArg(e.Args[0])}
		case "GoVersion.GreaterThan":
			return ir.FilterExpr{Op: ir.FilterGoVersionGreaterThanOp, Value: conv.parseStringArg(e.Args[0])}
		case "GoVersion.LessEqThan":
			return ir.FilterExpr{Op: ir.FilterGoVersionLessEqThanOp, Value: conv.parseStringArg(e.Args[0])}
		case "GoVersion.GreaterEqThan":
			return ir.FilterExpr{Op: ir.FilterGoVersionGreaterEqThanOp, Value: conv.parseStringArg(e.Args[0])}
		case "File.Imports":
			return ir.FilterExpr{Op: ir.FilterFileImportsOp, Value: conv.parseStringArg(e.Args[0])}
		case "File.PkgPath.Matches":
			return ir.FilterExpr{Op: ir.FilterFilePkgPathMatchesOp, Value: conv.parseStringArg(e.Args[0])}
		case "File.Name.Matches":
			return ir.FilterExpr{Op: ir.FilterFileNameMatchesOp, Value: conv.parseStringArg(e.Args[0])}

		case "Filter":
			funcName, ok := e.Args[0].(*ast.Ident)
			if !ok {
				panic(conv.errorf(e.Args[0], "only named function args are supported"))
			}
			args := []ir.FilterExpr{
				{Op: ir.FilterFilterFuncRefOp, Value: funcName.String()},
			}
			return ir.FilterExpr{Op: ir.FilterVarFilterOp, Value: op.varName, Args: args}
		}

		if macro := conv.findLocalMacro(e); macro != nil {
			return conv.expandMacro(macro, e)
		}

		args := convertExprList(e.Args)
		switch op.path {
		case "Value.Int":
			return ir.FilterExpr{Op: ir.FilterVarValueIntOp, Value: op.varName, Args: args}
		case "Text.Matches":
			return ir.FilterExpr{Op: ir.FilterVarTextMatchesOp, Value: op.varName, Args: args}
		case "Node.Is":
			return ir.FilterExpr{Op: ir.FilterVarNodeIsOp, Value: op.varName, Args: args}
		case "Node.Parent.Is":
			if op.varName != "$$" {
				// TODO: remove this restriction.
				panic(conv.errorf(e.Args[0], "only $$ parent nodes are implemented"))
			}
			return ir.FilterExpr{Op: ir.FilterRootNodeParentIsOp, Args: args}
		case "Object.Is":
			return ir.FilterExpr{Op: ir.FilterVarObjectIsOp, Value: op.varName, Args: args}
		case "Type.HasPointers":
			return ir.FilterExpr{Op: ir.FilterVarTypeHasPointersOp, Value: op.varName}
		case "Type.Is":
			return ir.FilterExpr{Op: ir.FilterVarTypeIsOp, Value: op.varName, Args: args}
		case "Type.Underlying.Is":
			return ir.FilterExpr{Op: ir.FilterVarTypeUnderlyingIsOp, Value: op.varName, Args: args}
		case "Type.OfKind":
			return ir.FilterExpr{Op: ir.FilterVarTypeOfKindOp, Value: op.varName, Args: args}
		case "Type.Underlying.OfKind":
			return ir.FilterExpr{Op: ir.FilterVarTypeUnderlyingOfKindOp, Value: op.varName, Args: args}
		case "Type.ConvertibleTo":
			return ir.FilterExpr{Op: ir.FilterVarTypeConvertibleToOp, Value: op.varName, Args: args}
		case "Type.AssignableTo":
			return ir.FilterExpr{Op: ir.FilterVarTypeAssignableToOp, Value: op.varName, Args: args}
		case "Type.Implements":
			return ir.FilterExpr{Op: ir.FilterVarTypeImplementsOp, Value: op.varName, Args: args}
		}
	}

	return ir.FilterExpr{}
}

func (conv *converter) parseStringArg(e ast.Expr) string {
	s, ok := conv.toStringValue(e)
	if !ok {
		panic(conv.errorf(e, "expected a string literal argument"))
	}
	return s
}

func (conv *converter) toStringValue(x ast.Node) (string, bool) {
	switch x := x.(type) {
	case *ast.BasicLit:
		if x.Kind != token.STRING {
			return "", false
		}
		s, err := strconv.Unquote(x.Value)
		if err != nil {
			return "", false
		}
		return s, true
	case ast.Expr:
		typ, ok := conv.types.Types[x]
		if !ok || typ.Type.String() != "string" {
			return "", false
		}
		str := constant.StringVal(typ.Value)
		return str, true
	}
	return "", false
}

func (conv *converter) inspectFilterSelector(e ast.Expr) filterExprSelector {
	var o filterExprSelector

	if call, ok := e.(*ast.CallExpr); ok {
		o.args = call.Args
		e = call.Fun
	}
	var path string
	for {
		if call, ok := e.(*ast.CallExpr); ok {
			e = call.Fun
			continue
		}
		selector, ok := e.(*ast.SelectorExpr)
		if !ok {
			break
		}
		if path == "" {
			path = selector.Sel.Name
		} else {
			path = selector.Sel.Name + "." + path
		}
		e = astutil.Unparen(selector.X)
	}

	o.path = path

	indexing, ok := astutil.Unparen(e).(*ast.IndexExpr)
	if !ok {
		return o
	}
	mapIdent, ok := astutil.Unparen(indexing.X).(*ast.Ident)
	if !ok {
		return o
	}
	o.mapName = mapIdent.Name
	indexString, _ := conv.toStringValue(indexing.Index)
	o.varName = indexString

	return o
}

type filterExprSelector struct {
	mapName string
	varName string
	path    string
	args    []ast.Expr
}
