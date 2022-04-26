// Copyright (c) 2017, Daniel Martí <mvdan@mvdan.cc>
// See LICENSE for licensing information

// Package check implements the unparam linter. Note that its API is not
// stable.
package check

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/constant"
	"go/parser"
	"go/printer"
	"go/token"
	"go/types"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// UnusedParams returns a list of human-readable issues that point out unused
// function parameters.
func UnusedParams(tests, exported, debug bool, args ...string) ([]string, error) {
	wd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	c := &Checker{
		wd:       wd,
		tests:    tests,
		exported: exported,
	}
	if debug {
		c.debugLog = os.Stderr
	}
	return c.lines(args...)
}

// Checker finds unused parameterss in a program. You probably want to use
// UnusedParams instead, unless you want to use a *loader.Program and
// *ssa.Program directly.
type Checker struct {
	pkgs []*packages.Package
	prog *ssa.Program

	wd string

	tests    bool
	exported bool
	debugLog io.Writer

	issues []Issue

	cachedDeclCounts map[string]map[string]int

	// callByPos maps from a call site position to its CallExpr.
	callByPos map[token.Pos]*ast.CallExpr

	// funcBodyByPos maps from a function position to its body. We can't map
	// to the declaration, as that could be either a FuncDecl or FuncLit.
	funcBodyByPos map[token.Pos]*ast.BlockStmt

	// typesImplementing records the method names that each named type needs
	// to typecheck properly, as they're required to implement interfaces.
	typesImplementing map[*types.Named][]string

	// localCallSites is a very simple form of a callgraph, only recording
	// direct function calls within a single package.
	localCallSites map[*ssa.Function][]ssa.CallInstruction

	// These three maps record whether an entire func's signature cannot be
	// changed, or only its list of parameters or results.

	signRequiredBy    map[*ssa.Function]string
	paramsRequiredBy  map[*ssa.Function]string
	resultsRequiredBy map[*ssa.Function]string
}

var errorType = types.Universe.Lookup("error").Type()

// lines runs the checker and returns the list of readable issues.
func (c *Checker) lines(args ...string) ([]string, error) {
	cfg := &packages.Config{
		Mode:  packages.LoadSyntax,
		Tests: c.tests,
	}
	pkgs, err := packages.Load(cfg, args...)
	if err != nil {
		return nil, err
	}
	if packages.PrintErrors(pkgs) > 0 {
		return nil, fmt.Errorf("encountered errors")
	}

	prog, _ := ssautil.Packages(pkgs, 0)
	prog.Build()
	c.Packages(pkgs)
	c.ProgramSSA(prog)
	issues, err := c.Check()
	if err != nil {
		return nil, err
	}
	lines := make([]string, 0, len(issues))
	prevLine := ""
	for _, issue := range issues {
		fpos := prog.Fset.Position(issue.Pos()).String()
		if strings.HasPrefix(fpos, c.wd) {
			fpos = fpos[len(c.wd)+1:]
		}
		line := fmt.Sprintf("%s: %s", fpos, issue.Message())
		if line == prevLine {
			// Deduplicate lines, since we may look at the same
			// package multiple times if tests are involved.
			// TODO: is there a better way to handle this?
			continue
		}
		prevLine = line
		lines = append(lines, fmt.Sprintf("%s: %s", fpos, issue.Message()))
	}
	return lines, nil
}

// Issue identifies a found unused parameter.
type Issue struct {
	pos   token.Pos
	fname string
	msg   string
}

func (i Issue) Pos() token.Pos  { return i.pos }
func (i Issue) Message() string { return i.fname + " - " + i.msg }

// Program supplies Checker with the needed *loader.Program.
func (c *Checker) Packages(pkgs []*packages.Package) {
	c.pkgs = pkgs
}

// ProgramSSA supplies Checker with the needed *ssa.Program.
func (c *Checker) ProgramSSA(prog *ssa.Program) {
	c.prog = prog
}

// CheckExportedFuncs sets whether to inspect exported functions
func (c *Checker) CheckExportedFuncs(exported bool) {
	c.exported = exported
}

func (c *Checker) debug(format string, a ...interface{}) {
	if c.debugLog != nil {
		fmt.Fprintf(c.debugLog, format, a...)
	}
}

// generatedDoc reports whether a comment text describes its file as being code
// generated.
func generatedDoc(text string) bool {
	return strings.Contains(text, "Code generated") ||
		strings.Contains(text, "DO NOT EDIT")
}

// eqlConsts reports whether two constant values, possibly nil, are equal.
func eqlConsts(c1, c2 *ssa.Const) bool {
	if c1 == nil || c2 == nil {
		return c1 == c2
	}
	if c1.Type() != c2.Type() {
		return false
	}
	if c1.Value == nil || c2.Value == nil {
		return c1.Value == c2.Value
	}
	return constant.Compare(c1.Value, token.EQL, c2.Value)
}

var stdSizes = types.SizesFor("gc", "amd64")

// Check runs the unused parameter check and returns the list of found issues,
// and any error encountered.
func (c *Checker) Check() ([]Issue, error) {
	c.cachedDeclCounts = make(map[string]map[string]int)
	c.callByPos = make(map[token.Pos]*ast.CallExpr)
	c.funcBodyByPos = make(map[token.Pos]*ast.BlockStmt)
	c.typesImplementing = make(map[*types.Named][]string)

	wantPkg := make(map[*types.Package]*packages.Package)
	genFiles := make(map[string]bool)
	for _, pkg := range c.pkgs {
		wantPkg[pkg.Types] = pkg
		for _, f := range pkg.Syntax {
			if len(f.Comments) > 0 && generatedDoc(f.Comments[0].Text()) {
				fname := c.prog.Fset.Position(f.Pos()).Filename
				genFiles[fname] = true
			}
			ast.Inspect(f, func(node ast.Node) bool {
				switch node := node.(type) {
				case *ast.ValueSpec:
					if len(node.Values) == 0 || node.Type == nil ||
						len(node.Names) != 1 || node.Names[0].Name != "_" {
						break
					}
					iface, ok := pkg.TypesInfo.TypeOf(node.Type).Underlying().(*types.Interface)
					if !ok {
						break
					}
					// var _ someIface = named
					valTyp := pkg.TypesInfo.Types[node.Values[0]].Type
					c.addImplementing(findNamed(valTyp), iface)
				case *ast.CallExpr:
					c.callByPos[node.Lparen] = node
				// ssa.Function.Pos returns the declaring
				// FuncLit.Type.Func or the position of the
				// FuncDecl.Name.
				case *ast.FuncDecl:
					c.funcBodyByPos[node.Name.Pos()] = node.Body
				case *ast.FuncLit:
					c.funcBodyByPos[node.Pos()] = node.Body
				}
				return true
			})
		}
	}
	allFuncs := ssautil.AllFunctions(c.prog)

	// map from *ssa.FreeVar to *ssa.Function, to find function literals
	// behind closure vars in the simpler scenarios.
	freeVars := map[*ssa.FreeVar]*ssa.Function{}
	for curFunc := range allFuncs {
		for _, b := range curFunc.Blocks {
			for _, instr := range b.Instrs {
				instr, ok := instr.(*ssa.MakeClosure)
				if !ok {
					continue
				}
				fn := instr.Fn.(*ssa.Function)
				for i, fv := range fn.FreeVars {
					binding := instr.Bindings[i]
					alloc, ok := binding.(*ssa.Alloc)
					if !ok {
						continue
					}
					for _, ref := range *alloc.Referrers() {
						store, ok := ref.(*ssa.Store)
						if !ok {
							continue
						}
						if fn, ok := store.Val.(*ssa.Function); ok {
							freeVars[fv] = fn
							break
						}
					}
				}
			}
		}
	}

	c.signRequiredBy = make(map[*ssa.Function]string)
	c.paramsRequiredBy = make(map[*ssa.Function]string)
	c.resultsRequiredBy = make(map[*ssa.Function]string)
	c.localCallSites = make(map[*ssa.Function][]ssa.CallInstruction)
	for curFunc := range allFuncs {
		if strings.HasPrefix(curFunc.Synthetic, "wrapper for func") {
			// Synthetic func wrappers are uninteresting, and can
			// lead to false negatives.
			continue
		}
		for _, b := range curFunc.Blocks {
			for _, instr := range b.Instrs {
				if instr, ok := instr.(ssa.CallInstruction); ok {
					if fn := findFunction(freeVars, instr.Common().Value); fn != nil {
						c.localCallSites[fn] = append(c.localCallSites[fn], instr)
					}
					fn := receivesExtractedArgs(freeVars, instr)
					if fn != nil {
						// fn(someFunc()) fixes params
						c.paramsRequiredBy[fn] = "forwarded call"
					}
				}
				switch instr := instr.(type) {
				case *ssa.Call:
					for _, arg := range instr.Call.Args {
						if fn := findFunction(freeVars, arg); fn != nil {
							// someFunc(fn)
							c.signRequiredBy[fn] = "call"
						}
					}
				case *ssa.Phi:
					for _, val := range instr.Edges {
						if fn := findFunction(freeVars, val); fn != nil {
							// nonConstVar = fn
							c.signRequiredBy[fn] = "phi"
						}
					}
				case *ssa.Return:
					for _, val := range instr.Results {
						if fn := findFunction(freeVars, val); fn != nil {
							// return fn
							c.signRequiredBy[fn] = "result"
						}
					}
					if call := callExtract(instr, instr.Results); call != nil {
						if fn := findFunction(freeVars, call.Call.Value); fn != nil {
							// return fn()
							c.resultsRequiredBy[fn] = "return"
						}
					}
				case *ssa.Store:
					as := ""
					switch instr.Addr.(type) {
					case *ssa.FieldAddr:
						// x.someField = fn
						as = "field"
					case *ssa.IndexAddr:
						// x[someIndex] = fn
						as = "element"
					case *ssa.Global:
						// someGlobal = fn
						as = "global"
					default:
						continue
					}
					if fn := findFunction(freeVars, instr.Val); fn != nil {
						c.signRequiredBy[fn] = as
					}
				case *ssa.MakeInterface:
					// someIface(named)
					iface := instr.Type().Underlying().(*types.Interface)
					c.addImplementing(findNamed(instr.X.Type()), iface)

					if fn := findFunction(freeVars, instr.X); fn != nil {
						// emptyIface = fn
						c.signRequiredBy[fn] = "interface"
					}
				case *ssa.ChangeType:
					if fn := findFunction(freeVars, instr.X); fn != nil {
						// someType(fn)
						c.signRequiredBy[fn] = "type conversion"
					}
				}
			}
		}
	}

	for fn := range allFuncs {
		switch {
		case fn.Pkg == nil: // builtin?
			continue
		case fn.Name() == "init":
			continue
		case len(fn.Blocks) == 0: // stub
			continue
		}
		pkg := wantPkg[fn.Pkg.Pkg]
		if pkg == nil { // not part of given pkgs
			continue
		}
		if c.exported || fn.Pkg.Pkg.Name() == "main" {
			// we want exported funcs, or this is a main package so
			// nothing is exported
		} else if strings.Contains(fn.Name(), "$") {
			// anonymous function within a possibly exported func
		} else if ast.IsExported(fn.Name()) {
			continue // user doesn't want to change signatures here
		}
		fname := c.prog.Fset.Position(fn.Pos()).Filename
		if genFiles[fname] {
			continue // generated file
		}

		c.checkFunc(fn, pkg)
	}
	sort.Slice(c.issues, func(i, j int) bool {
		p1 := c.prog.Fset.Position(c.issues[i].Pos())
		p2 := c.prog.Fset.Position(c.issues[j].Pos())
		if p1.Filename == p2.Filename {
			return p1.Offset < p2.Offset
		}
		return p1.Filename < p2.Filename
	})
	return c.issues, nil
}

func stringsContains(list []string, elem string) bool {
	for _, e := range list {
		if e == elem {
			return true
		}
	}
	return false
}

func (c *Checker) addImplementing(named *types.Named, iface *types.Interface) {
	if named == nil || iface == nil {
		return
	}
	list := c.typesImplementing[named]
	for i := 0; i < iface.NumMethods(); i++ {
		name := iface.Method(i).Name()
		if !stringsContains(list, name) {
			list = append(list, name)
		}
	}
	c.typesImplementing[named] = list
}

func findNamed(typ types.Type) *types.Named {
	switch typ := typ.(type) {
	case *types.Pointer:
		return findNamed(typ.Elem())
	case *types.Named:
		return typ
	}
	return nil
}

// findFunction returns the function that is behind a value, if any.
func findFunction(freeVars map[*ssa.FreeVar]*ssa.Function, value ssa.Value) *ssa.Function {
	switch value := value.(type) {
	case *ssa.Function:
		name := value.Name()
		if strings.HasSuffix(name, "$thunk") || strings.HasSuffix(name, "$bound") {
			// Method wrapper funcs contain a single block, which
			// calls the function being wrapped, and returns. We
			// want the function being wrapped.
			for _, instr := range value.Blocks[0].Instrs {
				call, ok := instr.(*ssa.Call)
				if !ok {
					continue
				}
				if callee := call.Call.StaticCallee(); callee != nil {
					return callee
				}
			}
			return nil // no static callee?
		}
		return value
	case *ssa.MakeClosure:
		// closure of a func
		return findFunction(freeVars, value.Fn)
	case *ssa.UnOp:
		if value.Op != token.MUL {
			break
		}
		if fv, ok := value.X.(*ssa.FreeVar); ok {
			return freeVars[fv]
		}
	}
	return nil
}

// addIssue records a newly found unused parameter.
func (c *Checker) addIssue(fn *ssa.Function, pos token.Pos, format string, args ...interface{}) {
	c.issues = append(c.issues, Issue{
		pos:   pos,
		fname: fn.RelString(fn.Package().Pkg),
		msg:   fmt.Sprintf(format, args...),
	})
}

// constValueString is cnst.Value.String() without panicking on untyped nils.
func constValueString(cnst *ssa.Const) string {
	if cnst.Value == nil {
		return "nil"
	}
	return cnst.Value.String()
}

// checkFunc checks a single function for unused parameters.
func (c *Checker) checkFunc(fn *ssa.Function, pkg *packages.Package) {
	c.debug("func %s\n", fn.RelString(fn.Package().Pkg))
	if dummyImpl(fn.Blocks[0]) { // panic implementation
		c.debug("  skip - dummy implementation\n")
		return
	}
	if by := c.signRequiredBy[fn]; by != "" {
		c.debug("  skip - func signature required by %s\n", by)
		return
	}
	if recv := fn.Signature.Recv(); recv != nil {
		named := findNamed(recv.Type())
		if stringsContains(c.typesImplementing[named], fn.Name()) {
			c.debug("  skip - method required to implement an interface\n")
			return
		}
	}
	if c.multipleImpls(pkg, fn) {
		c.debug("  skip - multiple implementations via build tags\n")
		return
	}
	paramsBy := c.paramsRequiredBy[fn]
	resultsBy := c.resultsRequiredBy[fn]
	callSites := c.localCallSites[fn]

	results := fn.Signature.Results()
	sameConsts := make([]*ssa.Const, results.Len())
	numRets := 0
	allRetsExtracting := true
	for _, block := range fn.Blocks {
		if resultsBy != "" {
			continue // we can't change the returns
		}
		last := block.Instrs[len(block.Instrs)-1]
		ret, ok := last.(*ssa.Return)
		if !ok {
			continue
		}
		for i, val := range ret.Results {
			if _, ok := val.(*ssa.Extract); !ok {
				allRetsExtracting = false
			}
			cnst := constValue(val)
			if numRets == 0 {
				sameConsts[i] = cnst
			} else if !eqlConsts(sameConsts[i], cnst) {
				sameConsts[i] = nil
			}
		}
		numRets++
	}
	for i, cnst := range sameConsts {
		if cnst == nil {
			// no consistent returned constant
			continue
		}
		if cnst.Value != nil && numRets == 1 {
			// just one return and it's not untyped nil (too many
			// false positives)
			continue
		}
		res := results.At(i)
		name := paramDesc(i, res)
		c.addIssue(fn, res.Pos(), "result %s is always %s", name, constValueString(cnst))
	}

resLoop:
	for i := 0; i < results.Len(); i++ {
		if resultsBy != "" {
			continue // we can't change the returns
		}
		if allRetsExtracting {
			continue
		}
		res := results.At(i)
		if res.Type() == errorType {
			// "error is never used" is less useful, and it's up to
			// tools like errcheck anyway.
			continue
		}
		count := 0
		for _, site := range callSites {
			val := site.Value()
			if val == nil { // e.g. go statement
				count++
				continue
			}
			for _, instr := range *val.Referrers() {
				extract, ok := instr.(*ssa.Extract)
				if !ok {
					continue resLoop // direct, real use
				}
				if extract.Index != i {
					continue // not the same result param
				}
				if len(*extract.Referrers()) > 0 {
					continue resLoop // real use after extraction
				}
			}
			count++
		}
		if count < 2 {
			continue // require ignoring at least twice
		}
		name := paramDesc(i, res)
		c.addIssue(fn, res.Pos(), "result %s is never used", name)
	}

	for i, par := range fn.Params {
		if paramsBy != "" {
			continue // we can't change the params
		}
		if i == 0 && fn.Signature.Recv() != nil { // receiver
			continue
		}
		c.debug("%s\n", par.String())
		switch par.Object().Name() {
		case "", "_": // unnamed
			c.debug("  skip - unnamed\n")
			continue
		}
		if stdSizes.Sizeof(par.Type()) == 0 {
			c.debug("  skip - zero size\n")
			continue
		}
		reason := "is unused"
		constStr := c.alwaysReceivedConst(callSites, par, i)
		if constStr != "" {
			reason = fmt.Sprintf("always receives %s", constStr)
		} else if c.anyRealUse(par, i, pkg) {
			c.debug("  skip - used somewhere in the func body\n")
			continue
		}
		c.addIssue(fn, par.Pos(), "%s %s", par.Name(), reason)
	}
}

// nodeStr stringifies a syntax tree node. It is only meant for simple nodes,
// such as short value expressions.
func nodeStr(node ast.Node) string {
	var buf bytes.Buffer
	fset := token.NewFileSet()
	if err := printer.Fprint(&buf, fset, node); err != nil {
		panic(err)
	}
	return buf.String()
}

// alwaysReceivedConst checks if a function parameter always receives the same
// constant value, given a list of inbound calls. If it does, a description of
// the value is returned. If not, an empty string is returned.
//
// This function is used to recommend that the parameter be replaced by a direct
// use of the constant. To avoid false positives, the function will return false
// if the number of inbound calls is too low.
func (c *Checker) alwaysReceivedConst(callSites []ssa.CallInstruction, par *ssa.Parameter, pos int) string {
	if len(callSites) < 4 {
		// We can't possibly receive the same constant value enough
		// times, hence a potential false positive.
		return ""
	}
	if ast.IsExported(par.Parent().Name()) {
		// we might not have all call sites for an exported func
		return ""
	}
	var seen *ssa.Const
	origPos := pos
	if par.Parent().Signature.Recv() != nil {
		// go/ast's CallExpr.Args does not include the receiver, but
		// go/ssa's equivalent does.
		origPos--
	}
	seenOrig := ""
	for _, site := range callSites {
		call := site.Common()
		if pos >= len(call.Args) {
			// TODO: investigate? Weird crash in
			// internal/x/net/http2/hpack/hpack_test.go, where we
			// roughly do: "at := d.mustAt; at(3)".
			return ""
		}
		cnst := constValue(call.Args[pos])
		if cnst == nil {
			return "" // not a constant
		}
		origArg := ""
		origCall := c.callByPos[call.Pos()]
		if origPos >= len(origCall.Args) {
			// variadic parameter that wasn't given
		} else {
			origArg = nodeStr(origCall.Args[origPos])
		}
		if seen == nil {
			seen = cnst // first constant
			seenOrig = origArg
		} else if !eqlConsts(seen, cnst) {
			return "" // different constants
		} else if origArg != seenOrig {
			seenOrig = ""
		}
	}
	seenStr := constValueString(seen)
	if seenOrig != "" && seenStr != seenOrig {
		return fmt.Sprintf("%s (%s)", seenOrig, seenStr)
	}
	return seenStr
}

func constValue(value ssa.Value) *ssa.Const {
	switch x := value.(type) {
	case *ssa.Const:
		return x
	case *ssa.MakeInterface:
		return constValue(x.X)
	}
	return nil
}

// anyRealUse reports whether a parameter has any relevant use within its
// function body. Certain uses are ignored, such as recursive calls where the
// parameter is re-used as itself.
func (c *Checker) anyRealUse(par *ssa.Parameter, pos int, pkg *packages.Package) bool {
	refs := *par.Referrers()
	if len(refs) == 0 {
		// Look for any uses like "_ = par", which are the developer's
		// way to tell they want to keep the parameter. SSA does not
		// keep that kind of statement around.
		body := c.funcBodyByPos[par.Parent().Pos()]
		any := false
		ast.Inspect(body, func(node ast.Node) bool {
			if any {
				return false
			}
			asgn, ok := node.(*ast.AssignStmt)
			if !ok || asgn.Tok != token.ASSIGN || len(asgn.Lhs) != 1 || len(asgn.Rhs) != 1 {
				return true
			}
			if left, ok := asgn.Lhs[0].(*ast.Ident); !ok || left.Name != "_" {
				return true
			}
			if right, ok := asgn.Rhs[0].(*ast.Ident); ok {
				obj := pkg.TypesInfo.Uses[right]
				if obj != nil && obj.Pos() == par.Pos() {
					any = true
				}
			}
			return true
		})
		return any
	}
refLoop:
	for _, ref := range refs {
		switch x := ref.(type) {
		case *ssa.Call:
			if x.Call.Value != par.Parent() {
				return true // not a recursive call
			}
			for i, arg := range x.Call.Args {
				if arg != par {
					continue
				}
				if i == pos {
					// reused directly in a recursive call
					continue refLoop
				}
			}
			return true
		case *ssa.Store:
			if insertedStore(x) {
				continue // inserted by go/ssa, not from the code
			}
			return true
		default:
			return true
		}
	}
	return false
}

// insertedStore reports whether a SSA instruction was inserted by the SSA
// building algorithm. That is, the store was not directly translated from an
// original Go statement.
func insertedStore(instr ssa.Instruction) bool {
	if instr.Pos() != token.NoPos {
		return false
	}
	store, ok := instr.(*ssa.Store)
	if !ok {
		return false
	}
	alloc, ok := store.Addr.(*ssa.Alloc)
	// we want exactly one use of this alloc value for it to be
	// inserted by ssa and dummy - the alloc instruction itself.
	return ok && len(*alloc.Referrers()) == 1
}

// rxHarmlessCall matches all the function expression strings which are allowed
// in a dummy implementation.
var rxHarmlessCall = regexp.MustCompile(`(?i)\b(log(ger)?|errors)\b|\bf?print|errorf?$`)

// dummyImpl reports whether a block is a dummy implementation. This is
// true if the block will almost immediately panic, throw or return
// constants only.
func dummyImpl(blk *ssa.BasicBlock) bool {
	var ops [8]*ssa.Value
	for _, instr := range blk.Instrs {
		if insertedStore(instr) {
			continue // inserted by go/ssa, not from the code
		}
		for _, val := range instr.Operands(ops[:0]) {
			switch x := (*val).(type) {
			case nil, *ssa.Const, *ssa.ChangeType, *ssa.Alloc,
				*ssa.MakeInterface, *ssa.MakeMap,
				*ssa.Function, *ssa.Global,
				*ssa.IndexAddr, *ssa.Slice,
				*ssa.UnOp, *ssa.Parameter:
			case *ssa.Call:
				if rxHarmlessCall.MatchString(x.Call.Value.String()) {
					continue
				}
			default:
				return false
			}
		}
		switch x := instr.(type) {
		case *ssa.Alloc, *ssa.Store, *ssa.UnOp, *ssa.BinOp,
			*ssa.MakeInterface, *ssa.MakeMap, *ssa.Extract,
			*ssa.IndexAddr, *ssa.FieldAddr, *ssa.Slice,
			*ssa.Lookup, *ssa.ChangeType, *ssa.TypeAssert,
			*ssa.Convert, *ssa.ChangeInterface:
			// non-trivial expressions in panic/log/print calls
		case *ssa.Return, *ssa.Panic:
			return true
		case *ssa.Call:
			if rxHarmlessCall.MatchString(x.Call.Value.String()) {
				continue
			}
			return x.Call.Value.Name() == "throw" // runtime's panic
		default:
			return false
		}
	}
	return false
}

// declCounts reports how many times a package's functions are declared. This is
// used, for example, to find if a function has many implementations.
//
// Since this function parses all of the package's Go source files on disk, its
// results are cached.
func (c *Checker) declCounts(pkgDir, pkgName string) map[string]int {
	key := pkgDir + ":" + pkgName
	if m, ok := c.cachedDeclCounts[key]; ok {
		return m
	}
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, pkgDir, nil, 0)
	if err != nil {
		// Don't panic or error here. In some part of the go/* libraries
		// stack, we sometimes end up with a package directory that is
		// wrong. That's not our fault, and we can't simply break the
		// tool until we fix the underlying issue.
		println(err.Error())
		c.cachedDeclCounts[pkgDir] = nil
		return nil
	}
	if len(pkgs) == 0 {
		// TODO: investigate why this started happening after switching
		// to go/packages
		return nil
	}
	pkg := pkgs[pkgName]
	count := make(map[string]int)
	for _, file := range pkg.Files {
		for _, decl := range file.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			name := recvPrefix(fd.Recv) + fd.Name.Name
			count[name]++
		}
	}
	c.cachedDeclCounts[key] = count
	return count
}

// recvPrefix returns the string prefix for a receiver field list. Star
// expressions are ignored, so as to conservatively assume that pointer and
// non-pointer receivers may still implement the same function.
//
// For example, for "function (*Foo) Bar()", recvPrefix will return "Foo.".
func recvPrefix(recv *ast.FieldList) string {
	if recv == nil {
		return ""
	}
	expr := recv.List[0].Type
	for {
		star, ok := expr.(*ast.StarExpr)
		if !ok {
			break
		}
		expr = star.X
	}
	id := expr.(*ast.Ident)
	return id.Name + "."
}

// multipleImpls reports whether a function has multiple implementations in the
// source code. For example, if there are different function bodies depending on
// the operating system or architecture. That tends to mean that an unused
// parameter in one implementation may not be unused in another.
func (c *Checker) multipleImpls(pkg *packages.Package, fn *ssa.Function) bool {
	if fn.Parent() != nil { // nested func
		return false
	}
	path := c.prog.Fset.Position(fn.Pos()).Filename
	count := c.declCounts(filepath.Dir(path), pkg.Types.Name())
	name := fn.Name()
	if recv := fn.Signature.Recv(); recv != nil {
		named := findNamed(recv.Type())
		name = named.Obj().Name() + "." + name
	}
	return count[name] > 1
}

// receivesExtractedArgs returns the statically called function, if its multiple
// arguments were all received via another function call. That is, if a call to
// function "foo" was of the form "foo(bar())". This often means that the
// parameters in "foo" are difficult to remove, even if unused.
func receivesExtractedArgs(freeVars map[*ssa.FreeVar]*ssa.Function, call ssa.CallInstruction) *ssa.Function {
	comm := call.Common()
	callee := findFunction(freeVars, comm.Value)
	if callee == nil {
		return nil
	}
	if callee.Signature.Params().Len() < 2 {
		// there aren't multiple parameters
		return nil
	}
	args := comm.Args
	if callee.Signature.Recv() != nil {
		// skip the receiver argument
		args = args[1:]
	}
	if c := callExtract(call, args); c != nil {
		return callee
	}
	return nil
}

// callExtract returns the call instruction fn(...) if it is used directly as
// arguments to the parent instruction, such as fn2(fn(...)) or return fn(...).
func callExtract(parent ssa.Instruction, values []ssa.Value) *ssa.Call {
	if len(values) == 1 {
		if call, ok := values[0].(*ssa.Call); ok {
			return call
		}
	}
	var prev *ssa.Call
	for i, val := range values {
		ext, ok := val.(*ssa.Extract)
		if !ok {
			return nil
		}
		if ext.Index != i {
			return nil // not extracted in the same order
		}
		call, ok := ext.Tuple.(*ssa.Call)
		if !ok {
			return nil // not a call
		}
		if prev == nil {
			prev = call
		} else if prev != call {
			return nil // not the same calls
		}
	}
	if prev == nil {
		return nil
	}
	if prev.Call.Signature().Results().Len() != len(values) {
		return nil // not extracting all the results
	}
	if prev.Pos() < parent.Pos() {
		// Of the form:
		//
		//   a, b := fn()
		//   fn2(a, b)
		return nil
	}
	return prev
}

// paramDesc returns a string describing a parameter variable. If the parameter
// had no name, the function will fall back to describing the parameter by its
// position within the parameter list and its type.
func paramDesc(i int, v *types.Var) string {
	name := v.Name()
	if name != "" && name != "_" {
		return name
	}
	return fmt.Sprintf("%d (%s)", i, v.Type().String())
}
