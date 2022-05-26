package ruleguard

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/constant"
	"go/parser"
	"go/token"
	"go/types"
	"io/ioutil"
	"regexp"

	"github.com/quasilyte/go-ruleguard/ruleguard/goutil"
	"github.com/quasilyte/go-ruleguard/ruleguard/ir"
	"github.com/quasilyte/go-ruleguard/ruleguard/quasigo"
	"github.com/quasilyte/go-ruleguard/ruleguard/textmatch"
	"github.com/quasilyte/go-ruleguard/ruleguard/typematch"
	"github.com/quasilyte/gogrep"
	"github.com/quasilyte/gogrep/nodetag"
)

type irLoaderConfig struct {
	ctx *LoadContext

	state *engineState

	importer *goImporter

	itab *typematch.ImportsTab

	pkg *types.Package

	gogrepFset *token.FileSet

	prefix      string
	importedPkg string
}

type irLoader struct {
	state *engineState
	ctx   *LoadContext
	itab  *typematch.ImportsTab

	pkg *types.Package

	file       *ir.File
	gogrepFset *token.FileSet

	filename string
	res      *goRuleSet

	importer *goImporter

	group *GoRuleGroup

	prefix      string // For imported packages, a prefix that is added to a rule group name
	importedPkg string // Package path; only for imported packages

	imported []*goRuleSet
}

func newIRLoader(config irLoaderConfig) *irLoader {
	return &irLoader{
		state:      config.state,
		ctx:        config.ctx,
		importer:   config.importer,
		itab:       config.itab,
		pkg:        config.pkg,
		prefix:     config.prefix,
		gogrepFset: config.gogrepFset,
	}
}

func (l *irLoader) LoadFile(filename string, f *ir.File) (*goRuleSet, error) {
	l.filename = filename
	l.file = f
	l.res = &goRuleSet{
		universal: &scopedGoRuleSet{},
		groups:    make(map[string]*GoRuleGroup),
	}

	for _, imp := range f.BundleImports {
		if l.importedPkg != "" {
			return nil, l.errorf(imp.Line, nil, "imports from imported packages are not supported yet")
		}
		if err := l.loadBundle(imp); err != nil {
			return nil, err
		}
	}

	if err := l.compileFilterFuncs(filename, f); err != nil {
		return nil, err
	}

	for i := range f.RuleGroups {
		if err := l.loadRuleGroup(&f.RuleGroups[i]); err != nil {
			return nil, err
		}
	}

	if len(l.imported) != 0 {
		toMerge := []*goRuleSet{l.res}
		toMerge = append(toMerge, l.imported...)
		merged, err := mergeRuleSets(toMerge)
		if err != nil {
			return nil, err
		}
		l.res = merged
	}

	return l.res, nil
}

func (l *irLoader) importErrorf(line int, wrapped error, format string, args ...interface{}) error {
	return &ImportError{
		msg: fmt.Sprintf("%s:%d: %s", l.filename, line, fmt.Sprintf(format, args...)),
		err: wrapped,
	}
}

func (l *irLoader) errorf(line int, wrapped error, format string, args ...interface{}) error {
	if wrapped == nil {
		return fmt.Errorf("%s:%d: %s", l.filename, line, fmt.Sprintf(format, args...))
	}
	return fmt.Errorf("%s:%d: %s: %w", l.filename, line, fmt.Sprintf(format, args...), wrapped)
}

func (l *irLoader) loadBundle(bundle ir.BundleImport) error {
	files, err := findBundleFiles(bundle.PkgPath)
	if err != nil {
		return l.errorf(bundle.Line, err, "can't find imported bundle files")
	}
	for _, filename := range files {
		rset, err := l.loadExternFile(bundle.Prefix, bundle.PkgPath, filename)
		if err != nil {
			return l.errorf(bundle.Line, err, "error during bundle file loading")
		}
		l.imported = append(l.imported, rset)
	}

	return nil
}

func (l *irLoader) loadExternFile(prefix, pkgPath, filename string) (*goRuleSet, error) {
	src, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	irfile, pkg, err := convertAST(l.ctx, l.importer, filename, src)
	if err != nil {
		return nil, err
	}
	config := irLoaderConfig{
		state:       l.state,
		ctx:         l.ctx,
		importer:    l.importer,
		prefix:      prefix,
		pkg:         pkg,
		importedPkg: pkgPath,
		itab:        l.itab,
		gogrepFset:  l.gogrepFset,
	}
	rset, err := newIRLoader(config).LoadFile(filename, irfile)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", l.importedPkg, err)
	}
	return rset, nil
}

func (l *irLoader) compileFilterFuncs(filename string, irfile *ir.File) error {
	if len(irfile.CustomDecls) == 0 {
		return nil
	}

	var buf bytes.Buffer
	buf.WriteString("package gorules\n")
	buf.WriteString("import \"github.com/quasilyte/go-ruleguard/dsl\"\n")
	buf.WriteString("import \"github.com/quasilyte/go-ruleguard/dsl/types\"\n")
	buf.WriteString("type _ = dsl.Matcher\n")
	buf.WriteString("type _ = types.Type\n")
	for _, src := range irfile.CustomDecls {
		buf.WriteString(src)
		buf.WriteString("\n")
	}

	fset := token.NewFileSet()
	f, err := goutil.LoadGoFile(goutil.LoadConfig{
		Fset:     fset,
		Filename: filename,
		Data:     &buf,
		Importer: l.importer,
	})
	if err != nil {
		// If this ever happens, user will get unexpected error
		// lines for it; but we should trust that 99.9% errors
		// should be catched at irconv phase so we get a valid Go
		// source here as well?
		return fmt.Errorf("parse custom decls: %w", err)
	}

	for _, decl := range f.Syntax.Decls {
		decl, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		ctx := &quasigo.CompileContext{
			Env:   l.state.env,
			Types: f.Types,
			Fset:  fset,
		}
		compiled, err := quasigo.Compile(ctx, decl)
		if err != nil {
			return err
		}
		if l.ctx.DebugFilter == decl.Name.String() {
			l.ctx.DebugPrint(quasigo.Disasm(l.state.env, compiled))
		}
		ctx.Env.AddFunc(f.Pkg.Path(), decl.Name.String(), compiled)
	}

	return nil
}

func (l *irLoader) loadRuleGroup(group *ir.RuleGroup) error {
	l.group = &GoRuleGroup{
		Line:       group.Line,
		Filename:   l.filename,
		Name:       group.Name,
		DocSummary: group.DocSummary,
		DocBefore:  group.DocBefore,
		DocAfter:   group.DocAfter,
		DocNote:    group.DocNote,
		DocTags:    group.DocTags,
	}
	if l.prefix != "" {
		l.group.Name = l.prefix + "/" + l.group.Name
	}

	if l.ctx.GroupFilter != nil && !l.ctx.GroupFilter(l.group.Name) {
		return nil // Skip this group
	}
	if _, ok := l.res.groups[l.group.Name]; ok {
		panic(fmt.Sprintf("duplicated function %s after the typecheck", l.group.Name)) // Should never happen
	}
	l.res.groups[l.group.Name] = l.group

	l.itab.EnterScope()
	defer l.itab.LeaveScope()

	for _, imported := range group.Imports {
		l.itab.Load(imported.Name, imported.Path)
	}

	for _, rule := range group.Rules {
		if err := l.loadRule(group, rule); err != nil {
			return err
		}
	}

	return nil
}

func (l *irLoader) loadRule(group *ir.RuleGroup, rule ir.Rule) error {
	proto := goRule{
		line:       rule.Line,
		group:      l.group,
		suggestion: rule.SuggestTemplate,
		msg:        rule.ReportTemplate,
		location:   rule.LocationVar,
	}

	info := filterInfo{
		Vars: make(map[string]struct{}),
	}
	if rule.WhereExpr.IsValid() {
		filter, err := l.newFilter(rule.WhereExpr, &info)
		if err != nil {
			return err
		}
		proto.filter = filter
	}

	for _, pat := range rule.SyntaxPatterns {
		if err := l.loadSyntaxRule(group, proto, info, rule, pat.Value, pat.Line); err != nil {
			return err
		}
	}
	for _, pat := range rule.CommentPatterns {
		if err := l.loadCommentRule(proto, rule, pat.Value, pat.Line); err != nil {
			return err
		}
	}
	return nil
}

func (l *irLoader) loadCommentRule(resultProto goRule, rule ir.Rule, src string, line int) error {
	dst := l.res.universal
	pat, err := regexp.Compile(src)
	if err != nil {
		return l.errorf(rule.Line, err, "compile regexp")
	}
	resultBase := resultProto
	resultBase.line = line
	result := goCommentRule{
		base:          resultProto,
		pat:           pat,
		captureGroups: regexpHasCaptureGroups(src),
	}
	dst.commentRules = append(dst.commentRules, result)

	return nil
}

func (l *irLoader) loadSyntaxRule(group *ir.RuleGroup, resultProto goRule, filterInfo filterInfo, rule ir.Rule, src string, line int) error {
	result := resultProto
	result.line = line

	var imports map[string]string
	if len(group.Imports) != 0 {
		imports = make(map[string]string)
		for _, imported := range group.Imports {
			imports[imported.Name] = imported.Path
		}
	}

	gogrepConfig := gogrep.CompileConfig{
		Fset:      l.gogrepFset,
		Src:       src,
		Strict:    false,
		WithTypes: true,
		Imports:   imports,
	}
	pat, info, err := gogrep.Compile(gogrepConfig)
	if err != nil {
		return l.errorf(rule.Line, err, "parse match pattern")
	}
	result.pat = pat

	for filterVar := range filterInfo.Vars {
		if filterVar == "$$" {
			continue // OK: a predefined var for the "entire match"
		}
		_, ok := info.Vars[filterVar]
		if !ok {
			return l.errorf(rule.Line, nil, "filter refers to a non-existing var %s", filterVar)
		}
	}

	dst := l.res.universal
	var dstTags []nodetag.Value
	switch tag := pat.NodeTag(); tag {
	case nodetag.Unknown:
		return l.errorf(rule.Line, nil, "can't infer a tag of %s", src)
	case nodetag.Node:
		return l.errorf(rule.Line, nil, "%s pattern is too general", src)
	case nodetag.StmtList:
		dstTags = []nodetag.Value{
			nodetag.BlockStmt,
			nodetag.CaseClause,
			nodetag.CommClause,
		}
	case nodetag.ExprList:
		dstTags = []nodetag.Value{
			nodetag.CallExpr,
			nodetag.CompositeLit,
			nodetag.ReturnStmt,
		}
	default:
		dstTags = []nodetag.Value{tag}
	}
	for _, tag := range dstTags {
		dst.rulesByTag[tag] = append(dst.rulesByTag[tag], result)
	}
	dst.categorizedNum++

	return nil
}

func (l *irLoader) unwrapTypeExpr(filter ir.FilterExpr) (types.Type, error) {
	typeString := l.unwrapStringExpr(filter)
	if typeString == "" {
		return nil, l.errorf(filter.Line, nil, "expected a non-empty type string")
	}
	typ, err := typeFromString(typeString)
	if err != nil {
		return nil, l.errorf(filter.Line, err, "parse type expr")
	}
	if typ == nil {
		return nil, l.errorf(filter.Line, nil, "can't convert %s into a type constraint yet", typeString)
	}
	return typ, nil
}

func (l *irLoader) unwrapInterfaceExpr(filter ir.FilterExpr) (*types.Interface, error) {
	typeString := l.unwrapStringExpr(filter)
	if typeString == "" {
		return nil, l.errorf(filter.Line, nil, "expected a non-empty type name string")
	}

	typ, err := l.state.FindType(l.importer, l.pkg, typeString)
	if err == nil {
		iface, ok := typ.Underlying().(*types.Interface)
		if !ok {
			return nil, l.errorf(filter.Line, nil, "%s is not an interface type", typeString)
		}
		return iface, nil
	}

	n, err := parser.ParseExpr(typeString)
	if err != nil {
		return nil, l.errorf(filter.Line, err, "parse %s type expr", typeString)
	}
	qn, ok := n.(*ast.SelectorExpr)
	if !ok {
		return nil, l.errorf(filter.Line, nil, "can't resolve %s type; try a fully-qualified name", typeString)
	}
	pkgName, ok := qn.X.(*ast.Ident)
	if !ok {
		return nil, l.errorf(filter.Line, nil, "invalid package name")
	}
	pkgPath, ok := l.itab.Lookup(pkgName.Name)
	if !ok {
		return nil, l.errorf(filter.Line, nil, "package %s is not imported", pkgName.Name)
	}
	pkg, err := l.importer.Import(pkgPath)
	if err != nil {
		return nil, l.importErrorf(filter.Line, err, "can't load %s", pkgPath)
	}
	obj := pkg.Scope().Lookup(qn.Sel.Name)
	if obj == nil {
		return nil, l.errorf(filter.Line, nil, "%s is not found in %s", qn.Sel.Name, pkgPath)
	}
	iface, ok := obj.Type().Underlying().(*types.Interface)
	if !ok {
		return nil, l.errorf(filter.Line, nil, "%s is not an interface type", qn.Sel.Name)
	}
	return iface, nil
}

func (l *irLoader) unwrapRegexpExpr(filter ir.FilterExpr) (textmatch.Pattern, error) {
	patternString := l.unwrapStringExpr(filter)
	if patternString == "" {
		return nil, l.errorf(filter.Line, nil, "expected a non-empty regexp pattern argument")
	}
	re, err := textmatch.Compile(patternString)
	if err != nil {
		return nil, l.errorf(filter.Line, err, "compile regexp")
	}
	return re, nil
}

func (l *irLoader) unwrapNodeTagExpr(filter ir.FilterExpr) (nodetag.Value, error) {
	typeString := l.unwrapStringExpr(filter)
	if typeString == "" {
		return nodetag.Unknown, l.errorf(filter.Line, nil, "expected a non-empty string argument")
	}
	tag := nodetag.FromString(typeString)
	if tag == nodetag.Unknown {
		return tag, l.errorf(filter.Line, nil, "%s is not a valid go/ast type name", typeString)
	}
	return tag, nil
}

func (l *irLoader) unwrapStringExpr(filter ir.FilterExpr) string {
	if filter.Op == ir.FilterStringOp {
		return filter.Value.(string)
	}
	return ""
}

func (l *irLoader) stringToBasicKind(s string) types.BasicInfo {
	switch s {
	case "integer":
		return types.IsInteger
	case "unsigned":
		return types.IsUnsigned
	case "float":
		return types.IsFloat
	case "complex":
		return types.IsComplex
	case "untyped":
		return types.IsUnsigned
	case "numeric":
		return types.IsNumeric
	default:
		return 0
	}
}

func (l *irLoader) newFilter(filter ir.FilterExpr, info *filterInfo) (matchFilter, error) {
	if filter.HasVar() {
		info.Vars[filter.Value.(string)] = struct{}{}
	}

	if filter.IsBinaryExpr() {
		return l.newBinaryExprFilter(filter, info)
	}

	result := matchFilter{src: filter.Src}

	switch filter.Op {
	case ir.FilterNotOp:
		x, err := l.newFilter(filter.Args[0], info)
		if err != nil {
			return result, err
		}
		result.fn = makeNotFilter(result.src, x)

	case ir.FilterVarTextMatchesOp:
		re, err := l.unwrapRegexpExpr(filter.Args[0])
		if err != nil {
			return result, err
		}
		result.fn = makeTextMatchesFilter(result.src, filter.Value.(string), re)

	case ir.FilterVarObjectIsOp:
		typeString := l.unwrapStringExpr(filter.Args[0])
		if typeString == "" {
			return result, l.errorf(filter.Line, nil, "expected a non-empty string argument")
		}
		switch typeString {
		case "Func", "Var", "Const", "TypeName", "Label", "PkgName", "Builtin", "Nil":
			// OK.
		default:
			return result, l.errorf(filter.Line, nil, "%s is not a valid go/types object name", typeString)
		}
		result.fn = makeObjectIsFilter(result.src, filter.Value.(string), typeString)

	case ir.FilterRootNodeParentIsOp:
		tag, err := l.unwrapNodeTagExpr(filter.Args[0])
		if err != nil {
			return result, err
		}
		result.fn = makeRootParentNodeIsFilter(result.src, tag)

	case ir.FilterVarNodeIsOp:
		tag, err := l.unwrapNodeTagExpr(filter.Args[0])
		if err != nil {
			return result, err
		}
		result.fn = makeNodeIsFilter(result.src, filter.Value.(string), tag)

	case ir.FilterVarTypeHasPointersOp:
		result.fn = makeTypeHasPointersFilter(result.src, filter.Value.(string))

	case ir.FilterVarTypeOfKindOp, ir.FilterVarTypeUnderlyingOfKindOp:
		kindString := l.unwrapStringExpr(filter.Args[0])
		if kindString == "" {
			return result, l.errorf(filter.Line, nil, "expected a non-empty string argument")
		}
		underlying := filter.Op == ir.FilterVarTypeUnderlyingOfKindOp
		switch kindString {
		case "signed":
			result.fn = makeTypeIsSignedFilter(result.src, filter.Value.(string), underlying)
		case "int":
			result.fn = makeTypeIsIntUintFilter(result.src, filter.Value.(string), underlying, types.Int)
		case "uint":
			result.fn = makeTypeIsIntUintFilter(result.src, filter.Value.(string), underlying, types.Uint)
		default:
			kind := l.stringToBasicKind(kindString)
			if kind == 0 {
				return result, l.errorf(filter.Line, nil, "unknown kind %s", kindString)
			}
			result.fn = makeTypeOfKindFilter(result.src, filter.Value.(string), underlying, kind)
		}

	case ir.FilterVarTypeIsOp, ir.FilterVarTypeUnderlyingIsOp:
		typeString := l.unwrapStringExpr(filter.Args[0])
		if typeString == "" {
			return result, l.errorf(filter.Line, nil, "expected a non-empty string argument")
		}
		ctx := typematch.Context{Itab: l.itab}
		pat, err := typematch.Parse(&ctx, typeString)
		if err != nil {
			return result, l.errorf(filter.Line, err, "parse type expr")
		}
		underlying := filter.Op == ir.FilterVarTypeUnderlyingIsOp
		result.fn = makeTypeIsFilter(result.src, filter.Value.(string), underlying, pat)

	case ir.FilterVarTypeConvertibleToOp:
		dstType, err := l.unwrapTypeExpr(filter.Args[0])
		if err != nil {
			return result, err
		}
		result.fn = makeTypeConvertibleToFilter(result.src, filter.Value.(string), dstType)

	case ir.FilterVarTypeAssignableToOp:
		dstType, err := l.unwrapTypeExpr(filter.Args[0])
		if err != nil {
			return result, err
		}
		result.fn = makeTypeAssignableToFilter(result.src, filter.Value.(string), dstType)

	case ir.FilterVarTypeImplementsOp:
		iface, err := l.unwrapInterfaceExpr(filter.Args[0])
		if err != nil {
			return result, err
		}
		result.fn = makeTypeImplementsFilter(result.src, filter.Value.(string), iface)

	case ir.FilterVarPureOp:
		result.fn = makePureFilter(result.src, filter.Value.(string))
	case ir.FilterVarConstOp:
		result.fn = makeConstFilter(result.src, filter.Value.(string))
	case ir.FilterVarConstSliceOp:
		result.fn = makeConstSliceFilter(result.src, filter.Value.(string))
	case ir.FilterVarAddressableOp:
		result.fn = makeAddressableFilter(result.src, filter.Value.(string))

	case ir.FilterFileImportsOp:
		result.fn = makeFileImportsFilter(result.src, filter.Value.(string))

	case ir.FilterDeadcodeOp:
		result.fn = makeDeadcodeFilter(result.src)

	case ir.FilterGoVersionEqOp:
		version, err := ParseGoVersion(filter.Value.(string))
		if err != nil {
			return result, l.errorf(filter.Line, err, "parse Go version")
		}
		result.fn = makeGoVersionFilter(result.src, token.EQL, version)
	case ir.FilterGoVersionLessThanOp:
		version, err := ParseGoVersion(filter.Value.(string))
		if err != nil {
			return result, l.errorf(filter.Line, err, "parse Go version")
		}
		result.fn = makeGoVersionFilter(result.src, token.LSS, version)
	case ir.FilterGoVersionGreaterThanOp:
		version, err := ParseGoVersion(filter.Value.(string))
		if err != nil {
			return result, l.errorf(filter.Line, err, "parse Go version")
		}
		result.fn = makeGoVersionFilter(result.src, token.GTR, version)
	case ir.FilterGoVersionLessEqThanOp:
		version, err := ParseGoVersion(filter.Value.(string))
		if err != nil {
			return result, l.errorf(filter.Line, err, "parse Go version")
		}
		result.fn = makeGoVersionFilter(result.src, token.LEQ, version)
	case ir.FilterGoVersionGreaterEqThanOp:
		version, err := ParseGoVersion(filter.Value.(string))
		if err != nil {
			return result, l.errorf(filter.Line, err, "parse Go version")
		}
		result.fn = makeGoVersionFilter(result.src, token.GEQ, version)

	case ir.FilterFilePkgPathMatchesOp:
		re, err := regexp.Compile(filter.Value.(string))
		if err != nil {
			return result, l.errorf(filter.Line, err, "compile regexp")
		}
		result.fn = makeFilePkgPathMatchesFilter(result.src, re)

	case ir.FilterFileNameMatchesOp:
		re, err := regexp.Compile(filter.Value.(string))
		if err != nil {
			return result, l.errorf(filter.Line, err, "compile regexp")
		}
		result.fn = makeFileNameMatchesFilter(result.src, re)

	case ir.FilterVarFilterOp:
		funcName := filter.Args[0].Value.(string)
		userFn := l.state.env.GetFunc(l.file.PkgPath, funcName)
		if userFn == nil {
			return result, l.errorf(filter.Line, nil, "can't find a compiled version of %s", funcName)
		}
		result.fn = makeCustomVarFilter(result.src, filter.Value.(string), userFn)
	}

	if result.fn == nil {
		return result, l.errorf(filter.Line, nil, "unsupported expr: %s (%s)", result.src, filter.Op)
	}

	return result, nil
}

func (l *irLoader) newBinaryExprFilter(filter ir.FilterExpr, info *filterInfo) (matchFilter, error) {
	if filter.Op == ir.FilterAndOp || filter.Op == ir.FilterOrOp {
		result := matchFilter{src: filter.Src}
		lhs, err := l.newFilter(filter.Args[0], info)
		if err != nil {
			return result, err
		}
		rhs, err := l.newFilter(filter.Args[1], info)
		if err != nil {
			return result, err
		}
		if filter.Op == ir.FilterAndOp {
			result.fn = makeAndFilter(lhs, rhs)
		} else {
			result.fn = makeOrFilter(lhs, rhs)
		}
		return result, nil
	}

	// If constexpr is on the LHS, move it to the right, so the code below
	// can imply constants being on the RHS all the time.
	if filter.Args[0].IsBasicLit() && !filter.Args[1].IsBasicLit() {
		// Just a precaution: if we ever have a float values here,
		// we may not want to rearrange anything.
		switch filter.Args[0].Value.(type) {
		case string, int64:
			switch filter.Op {
			case ir.FilterEqOp, ir.FilterNeqOp:
				// Simple commutative ops. Just swap the args.
				newFilter := filter
				newFilter.Args = []ir.FilterExpr{filter.Args[1], filter.Args[0]}
				return l.newBinaryExprFilter(newFilter, info)
			}
		}
	}

	result := matchFilter{src: filter.Src}

	var tok token.Token
	switch filter.Op {
	case ir.FilterEqOp:
		tok = token.EQL
	case ir.FilterNeqOp:
		tok = token.NEQ
	case ir.FilterGtOp:
		tok = token.GTR
	case ir.FilterGtEqOp:
		tok = token.GEQ
	case ir.FilterLtOp:
		tok = token.LSS
	case ir.FilterLtEqOp:
		tok = token.LEQ
	default:
		return result, l.errorf(filter.Line, nil, "unsupported operator in binary expr: %s", result.src)
	}

	lhs := filter.Args[0]
	rhs := filter.Args[1]
	var rhsValue constant.Value
	switch rhs.Op {
	case ir.FilterStringOp:
		rhsValue = constant.MakeString(rhs.Value.(string))
	case ir.FilterIntOp:
		rhsValue = constant.MakeInt64(rhs.Value.(int64))
	}

	switch lhs.Op {
	case ir.FilterVarLineOp:
		if rhsValue != nil {
			result.fn = makeLineConstFilter(result.src, lhs.Value.(string), tok, rhsValue)
		} else if rhs.Op == lhs.Op {
			result.fn = makeLineFilter(result.src, lhs.Value.(string), tok, rhs.Value.(string))
		}
	case ir.FilterVarTypeSizeOp:
		if rhsValue != nil {
			result.fn = makeTypeSizeConstFilter(result.src, lhs.Value.(string), tok, rhsValue)
		}
	case ir.FilterVarValueIntOp:
		if rhsValue != nil {
			result.fn = makeValueIntConstFilter(result.src, lhs.Value.(string), tok, rhsValue)
		} else if rhs.Op == lhs.Op {
			result.fn = makeValueIntFilter(result.src, lhs.Value.(string), tok, rhs.Value.(string))
		}
	case ir.FilterVarTextOp:
		if rhsValue != nil {
			result.fn = makeTextConstFilter(result.src, lhs.Value.(string), tok, rhsValue)
		} else if rhs.Op == lhs.Op {
			result.fn = makeTextFilter(result.src, lhs.Value.(string), tok, rhs.Value.(string))
		}
	}

	if result.fn == nil {
		return result, l.errorf(filter.Line, nil, "unsupported binary expr: %s", result.src)
	}

	return result, nil
}

type filterInfo struct {
	Vars map[string]struct{}
}
