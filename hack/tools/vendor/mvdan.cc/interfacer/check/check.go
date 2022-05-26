// Copyright (c) 2015, Daniel Martí <mvdan@mvdan.cc>
// See LICENSE for licensing information

package check // import "mvdan.cc/interfacer/check"

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"strings"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"

	"github.com/kisielk/gotool"
	"mvdan.cc/lint"
)

func toDiscard(usage *varUsage) bool {
	if usage.discard {
		return true
	}
	for to := range usage.assigned {
		if toDiscard(to) {
			return true
		}
	}
	return false
}

func allCalls(usage *varUsage, all, ftypes map[string]string) {
	for fname := range usage.calls {
		all[fname] = ftypes[fname]
	}
	for to := range usage.assigned {
		allCalls(to, all, ftypes)
	}
}

func (c *Checker) interfaceMatching(param *types.Var, usage *varUsage) (string, string) {
	if toDiscard(usage) {
		return "", ""
	}
	ftypes := typeFuncMap(param.Type())
	called := make(map[string]string, len(usage.calls))
	allCalls(usage, called, ftypes)
	s := funcMapString(called)
	return c.ifaces[s], s
}

type varUsage struct {
	calls   map[string]struct{}
	discard bool

	assigned map[*varUsage]struct{}
}

type funcDecl struct {
	astDecl *ast.FuncDecl
	ssaFn   *ssa.Function
}

// CheckArgs checks the packages specified by their import paths in
// args.
func CheckArgs(args []string) ([]string, error) {
	paths := gotool.ImportPaths(args)
	conf := loader.Config{}
	conf.AllowErrors = true
	rest, err := conf.FromArgs(paths, false)
	if err != nil {
		return nil, err
	}
	if len(rest) > 0 {
		return nil, fmt.Errorf("unwanted extra args: %v", rest)
	}
	lprog, err := conf.Load()
	if err != nil {
		return nil, err
	}
	prog := ssautil.CreateProgram(lprog, 0)
	prog.Build()
	c := new(Checker)
	c.Program(lprog)
	c.ProgramSSA(prog)
	issues, err := c.Check()
	if err != nil {
		return nil, err
	}
	wd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	lines := make([]string, len(issues))
	for i, issue := range issues {
		fpos := prog.Fset.Position(issue.Pos()).String()
		if strings.HasPrefix(fpos, wd) {
			fpos = fpos[len(wd)+1:]
		}
		lines[i] = fmt.Sprintf("%s: %s", fpos, issue.Message())
	}
	return lines, nil
}

type Checker struct {
	lprog *loader.Program
	prog  *ssa.Program

	pkgTypes
	*loader.PackageInfo

	funcs []*funcDecl

	ssaByPos map[token.Pos]*ssa.Function

	discardFuncs map[*types.Signature]struct{}

	vars map[*types.Var]*varUsage
}

var (
	_ lint.Checker = (*Checker)(nil)
	_ lint.WithSSA = (*Checker)(nil)
)

func (c *Checker) Program(lprog *loader.Program) {
	c.lprog = lprog
}

func (c *Checker) ProgramSSA(prog *ssa.Program) {
	c.prog = prog
}

func (c *Checker) Check() ([]lint.Issue, error) {
	var total []lint.Issue
	c.ssaByPos = make(map[token.Pos]*ssa.Function)
	wantPkg := make(map[*types.Package]bool)
	for _, pinfo := range c.lprog.InitialPackages() {
		wantPkg[pinfo.Pkg] = true
	}
	for fn := range ssautil.AllFunctions(c.prog) {
		if fn.Pkg == nil { // builtin?
			continue
		}
		if len(fn.Blocks) == 0 { // stub
			continue
		}
		if !wantPkg[fn.Pkg.Pkg] { // not part of given pkgs
			continue
		}
		c.ssaByPos[fn.Pos()] = fn
	}
	for _, pinfo := range c.lprog.InitialPackages() {
		pkg := pinfo.Pkg
		c.getTypes(pkg)
		c.PackageInfo = c.lprog.AllPackages[pkg]
		total = append(total, c.checkPkg()...)
	}
	return total, nil
}

func (c *Checker) checkPkg() []lint.Issue {
	c.discardFuncs = make(map[*types.Signature]struct{})
	c.vars = make(map[*types.Var]*varUsage)
	c.funcs = c.funcs[:0]
	findFuncs := func(node ast.Node) bool {
		decl, ok := node.(*ast.FuncDecl)
		if !ok {
			return true
		}
		ssaFn := c.ssaByPos[decl.Name.Pos()]
		if ssaFn == nil {
			return true
		}
		fd := &funcDecl{
			astDecl: decl,
			ssaFn:   ssaFn,
		}
		if c.funcSigns[signString(fd.ssaFn.Signature)] {
			// implements interface
			return true
		}
		c.funcs = append(c.funcs, fd)
		ast.Walk(c, decl.Body)
		return true
	}
	for _, f := range c.Files {
		ast.Inspect(f, findFuncs)
	}
	return c.packageIssues()
}

func paramVarAndType(sign *types.Signature, i int) (*types.Var, types.Type) {
	params := sign.Params()
	extra := sign.Variadic() && i >= params.Len()-1
	if !extra {
		if i >= params.Len() {
			// builtins with multiple signatures
			return nil, nil
		}
		vr := params.At(i)
		return vr, vr.Type()
	}
	last := params.At(params.Len() - 1)
	switch x := last.Type().(type) {
	case *types.Slice:
		return nil, x.Elem()
	default:
		return nil, x
	}
}

func (c *Checker) varUsage(e ast.Expr) *varUsage {
	id, ok := e.(*ast.Ident)
	if !ok {
		return nil
	}
	param, ok := c.ObjectOf(id).(*types.Var)
	if !ok {
		// not a variable
		return nil
	}
	if usage, e := c.vars[param]; e {
		return usage
	}
	if !interesting(param.Type()) {
		return nil
	}
	usage := &varUsage{
		calls:    make(map[string]struct{}),
		assigned: make(map[*varUsage]struct{}),
	}
	c.vars[param] = usage
	return usage
}

func (c *Checker) addUsed(e ast.Expr, as types.Type) {
	if as == nil {
		return
	}
	if usage := c.varUsage(e); usage != nil {
		// using variable
		iface, ok := as.Underlying().(*types.Interface)
		if !ok {
			usage.discard = true
			return
		}
		for i := 0; i < iface.NumMethods(); i++ {
			m := iface.Method(i)
			usage.calls[m.Name()] = struct{}{}
		}
	} else if t, ok := c.TypeOf(e).(*types.Signature); ok {
		// using func
		c.discardFuncs[t] = struct{}{}
	}
}

func (c *Checker) addAssign(to, from ast.Expr) {
	pto := c.varUsage(to)
	pfrom := c.varUsage(from)
	if pto == nil || pfrom == nil {
		// either isn't interesting
		return
	}
	pfrom.assigned[pto] = struct{}{}
}

func (c *Checker) discard(e ast.Expr) {
	if usage := c.varUsage(e); usage != nil {
		usage.discard = true
	}
}

func (c *Checker) comparedWith(e, with ast.Expr) {
	if _, ok := with.(*ast.BasicLit); ok {
		c.discard(e)
	}
}

func (c *Checker) Visit(node ast.Node) ast.Visitor {
	switch x := node.(type) {
	case *ast.SelectorExpr:
		if _, ok := c.TypeOf(x.Sel).(*types.Signature); !ok {
			c.discard(x.X)
		}
	case *ast.StarExpr:
		c.discard(x.X)
	case *ast.UnaryExpr:
		c.discard(x.X)
	case *ast.IndexExpr:
		c.discard(x.X)
	case *ast.IncDecStmt:
		c.discard(x.X)
	case *ast.BinaryExpr:
		switch x.Op {
		case token.EQL, token.NEQ:
			c.comparedWith(x.X, x.Y)
			c.comparedWith(x.Y, x.X)
		default:
			c.discard(x.X)
			c.discard(x.Y)
		}
	case *ast.ValueSpec:
		for _, val := range x.Values {
			c.addUsed(val, c.TypeOf(x.Type))
		}
	case *ast.AssignStmt:
		for i, val := range x.Rhs {
			left := x.Lhs[i]
			if x.Tok == token.ASSIGN {
				c.addUsed(val, c.TypeOf(left))
			}
			c.addAssign(left, val)
		}
	case *ast.CompositeLit:
		for i, e := range x.Elts {
			switch y := e.(type) {
			case *ast.KeyValueExpr:
				c.addUsed(y.Key, c.TypeOf(y.Value))
				c.addUsed(y.Value, c.TypeOf(y.Key))
			case *ast.Ident:
				c.addUsed(y, compositeIdentType(c.TypeOf(x), i))
			}
		}
	case *ast.CallExpr:
		switch y := c.TypeOf(x.Fun).Underlying().(type) {
		case *types.Signature:
			c.onMethodCall(x, y)
		default:
			// type conversion
			if len(x.Args) == 1 {
				c.addUsed(x.Args[0], y)
			}
		}
	}
	return c
}

func compositeIdentType(t types.Type, i int) types.Type {
	switch x := t.(type) {
	case *types.Named:
		return compositeIdentType(x.Underlying(), i)
	case *types.Struct:
		return x.Field(i).Type()
	case *types.Array:
		return x.Elem()
	case *types.Slice:
		return x.Elem()
	}
	return nil
}

func (c *Checker) onMethodCall(ce *ast.CallExpr, sign *types.Signature) {
	for i, e := range ce.Args {
		paramObj, t := paramVarAndType(sign, i)
		// Don't if this is a parameter being re-used as itself
		// in a recursive call
		if id, ok := e.(*ast.Ident); ok {
			if paramObj == c.ObjectOf(id) {
				continue
			}
		}
		c.addUsed(e, t)
	}
	sel, ok := ce.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	// receiver func call on the left side
	if usage := c.varUsage(sel.X); usage != nil {
		usage.calls[sel.Sel.Name] = struct{}{}
	}
}

func (fd *funcDecl) paramGroups() [][]*types.Var {
	astList := fd.astDecl.Type.Params.List
	groups := make([][]*types.Var, len(astList))
	signIndex := 0
	for i, field := range astList {
		group := make([]*types.Var, len(field.Names))
		for j := range field.Names {
			group[j] = fd.ssaFn.Signature.Params().At(signIndex)
			signIndex++
		}
		groups[i] = group
	}
	return groups
}

func (c *Checker) packageIssues() []lint.Issue {
	var issues []lint.Issue
	for _, fd := range c.funcs {
		if _, e := c.discardFuncs[fd.ssaFn.Signature]; e {
			continue
		}
		for _, group := range fd.paramGroups() {
			issues = append(issues, c.groupIssues(fd, group)...)
		}
	}
	return issues
}

type Issue struct {
	pos token.Pos
	msg string
}

func (i Issue) Pos() token.Pos  { return i.pos }
func (i Issue) Message() string { return i.msg }

func (c *Checker) groupIssues(fd *funcDecl, group []*types.Var) []lint.Issue {
	var issues []lint.Issue
	for _, param := range group {
		usage := c.vars[param]
		if usage == nil {
			return nil
		}
		newType := c.paramNewType(fd.astDecl.Name.Name, param, usage)
		if newType == "" {
			return nil
		}
		issues = append(issues, Issue{
			pos: param.Pos(),
			msg: fmt.Sprintf("%s can be %s", param.Name(), newType),
		})
	}
	return issues
}

func willAddAllocation(t types.Type) bool {
	switch t.Underlying().(type) {
	case *types.Pointer, *types.Interface:
		return false
	}
	return true
}

func (c *Checker) paramNewType(funcName string, param *types.Var, usage *varUsage) string {
	t := param.Type()
	if !ast.IsExported(funcName) && willAddAllocation(t) {
		return ""
	}
	if named := typeNamed(t); named != nil {
		tname := named.Obj().Name()
		vname := param.Name()
		if mentionsName(funcName, tname) || mentionsName(funcName, vname) {
			return ""
		}
	}
	ifname, iftype := c.interfaceMatching(param, usage)
	if ifname == "" {
		return ""
	}
	if types.IsInterface(t.Underlying()) {
		if have := funcMapString(typeFuncMap(t)); have == iftype {
			return ""
		}
	}
	return ifname
}
