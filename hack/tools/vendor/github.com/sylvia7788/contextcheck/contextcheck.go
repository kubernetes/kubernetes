package contextcheck

import (
	"go/ast"
	"go/token"
	"go/types"
	"strconv"
	"strings"
	"sync"

	"github.com/gostaticanalysis/analysisutil"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"
	"golang.org/x/tools/go/ssa"
)

func NewAnalyzer() *analysis.Analyzer {
	return &analysis.Analyzer{
		Name: "contextcheck",
		Doc:  "check the function whether use a non-inherited context",
		Run:  NewRun(),
		Requires: []*analysis.Analyzer{
			buildssa.Analyzer,
		},
	}
}

const (
	ctxPkg  = "context"
	ctxName = "Context"
)

const (
	CtxIn      int = 1 << iota // ctx in function's param
	CtxOut                     // ctx in function's results
	CtxInField                 // ctx in function's field param

	CtxInOut = CtxIn | CtxOut
)

var (
	checkedMap     = make(map[string]bool)
	checkedMapLock sync.RWMutex
)

type runner struct {
	pass     *analysis.Pass
	ctxTyp   *types.Named
	ctxPTyp  *types.Pointer
	cmpPath  string
	skipFile map[*ast.File]bool
}

func NewRun() func(pass *analysis.Pass) (interface{}, error) {
	return func(pass *analysis.Pass) (interface{}, error) {
		r := new(runner)
		r.run(pass)
		return nil, nil
	}
}

func (r *runner) run(pass *analysis.Pass) {
	r.pass = pass
	r.cmpPath = strings.Split(pass.Pkg.Path(), "/")[0]
	pssa := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA)
	funcs := pssa.SrcFuncs
	name := pass.Pkg.Path()
	_ = name

	pkg := pssa.Pkg.Prog.ImportedPackage(ctxPkg)
	if pkg == nil {
		return
	}

	ctxType := pkg.Type(ctxName)
	if ctxType == nil {
		return
	}

	if resNamed, ok := ctxType.Object().Type().(*types.Named); !ok {
		return
	} else {
		r.ctxTyp = resNamed
		r.ctxPTyp = types.NewPointer(resNamed)
	}

	r.skipFile = make(map[*ast.File]bool)

	for _, f := range funcs {
		// skip checked function
		key := f.RelString(nil)
		_, ok := getValue(key)
		if ok {
			continue
		}

		if !r.checkIsEntry(f, f.Pos()) {
			continue
		}

		r.checkFuncWithCtx(f)
		setValue(key, true)
	}
}

func (r *runner) noImportedContext(f *ssa.Function) (ret bool) {
	if !f.Pos().IsValid() {
		return false
	}

	file := analysisutil.File(r.pass, f.Pos())
	if file == nil {
		return false
	}

	if skip, has := r.skipFile[file]; has {
		return skip
	}
	defer func() {
		r.skipFile[file] = ret
	}()

	for _, impt := range file.Imports {
		path, err := strconv.Unquote(impt.Path.Value)
		if err != nil {
			continue
		}
		path = analysisutil.RemoveVendor(path)
		if path == ctxPkg {
			return false
		}
	}

	return true
}

func (r *runner) checkIsEntry(f *ssa.Function, pos token.Pos) (ret bool) {
	if r.noImportedContext(f) {
		return false
	}

	// check params
	tuple := f.Signature.Params()
	for i := 0; i < tuple.Len(); i++ {
		if r.isCtxType(tuple.At(i).Type()) {
			ret = true
			break
		}
	}

	// check freevars
	for _, param := range f.FreeVars {
		if r.isCtxType(param.Type()) {
			ret = true
			break
		}
	}

	// check results
	tuple = f.Signature.Results()
	for i := 0; i < tuple.Len(); i++ {
		// skip the function which generate ctx
		if r.isCtxType(tuple.At(i).Type()) {
			ret = false
			break
		}
	}

	return
}

func (r *runner) collectCtxRef(f *ssa.Function) (refMap map[ssa.Instruction]bool, ok bool) {
	ok = true
	refMap = make(map[ssa.Instruction]bool)
	checkedRefMap := make(map[ssa.Value]bool)
	storeInstrs := make(map[*ssa.Store]bool)
	phiInstrs := make(map[*ssa.Phi]bool)

	var checkRefs func(val ssa.Value, fromAddr bool)
	var checkInstr func(instr ssa.Instruction, fromAddr bool)

	checkRefs = func(val ssa.Value, fromAddr bool) {
		if val == nil || val.Referrers() == nil {
			return
		}

		if checkedRefMap[val] {
			return
		}
		checkedRefMap[val] = true

		for _, instr := range *val.Referrers() {
			checkInstr(instr, fromAddr)
		}
	}

	checkInstr = func(instr ssa.Instruction, fromAddr bool) {
		switch i := instr.(type) {
		case ssa.CallInstruction:
			refMap[i] = true
			tp := r.getCallInstrCtxType(i)
			if tp&CtxOut != 0 {
				// collect referrers of the results
				checkRefs(i.Value(), false)
				return
			}
		case *ssa.Store:
			if fromAddr {
				// collect all store to judge whether it's right value is valid
				storeInstrs[i] = true
			} else {
				checkRefs(i.Addr, true)
			}
		case *ssa.UnOp:
			checkRefs(i, false)
		case *ssa.MakeClosure:
			for _, param := range i.Bindings {
				if r.isCtxType(param.Type()) {
					refMap[i] = true
					break
				}
			}
		case *ssa.Extract:
			// only care about ctx
			if r.isCtxType(i.Type()) {
				checkRefs(i, false)
			}
		case *ssa.Phi:
			phiInstrs[i] = true
			checkRefs(i, false)
		case *ssa.TypeAssert:
			// ctx.(*bm.Context)
		}
	}

	for _, param := range f.Params {
		if r.isCtxType(param.Type()) {
			checkRefs(param, false)
		}
	}

	for _, param := range f.FreeVars {
		if r.isCtxType(param.Type()) {
			checkRefs(param, false)
		}
	}

	for instr := range storeInstrs {
		if !checkedRefMap[instr.Val] {
			r.pass.Reportf(instr.Pos(), "Non-inherited new context, use function like `context.WithXXX` instead")
			ok = false
		}
	}

	for instr := range phiInstrs {
		for _, v := range instr.Edges {
			if !checkedRefMap[v] {
				r.pass.Reportf(instr.Pos(), "Non-inherited new context, use function like `context.WithXXX` instead")
				ok = false
			}
		}
	}

	return
}

func (r *runner) buildPkg(f *ssa.Function) {
	if f.Blocks != nil {
		return
	}

	// only build the pkg which is in the same repo
	if r.checkIsSameRepo(f.Pkg.Pkg.Path()) {
		f.Pkg.Build()
	}
}

func (r *runner) checkIsSameRepo(s string) bool {
	return strings.HasPrefix(s, r.cmpPath+"/")
}

func (r *runner) checkFuncWithCtx(f *ssa.Function) {
	refMap, ok := r.collectCtxRef(f)
	if !ok {
		return
	}

	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			tp, ok := r.getCtxType(instr)
			if !ok {
				continue
			}

			// checked in collectCtxRef, skipped
			if tp&CtxOut != 0 {
				continue
			}

			if tp&CtxIn != 0 {
				if !refMap[instr] {
					r.pass.Reportf(instr.Pos(), "Non-inherited new context, use function like `context.WithXXX` instead")
				}
			}

			ff := r.getFunction(instr)
			if ff == nil {
				continue
			}

			key := ff.RelString(nil)
			valid, ok := getValue(key)
			if ok {
				if !valid {
					r.pass.Reportf(instr.Pos(), "Function `%s` should pass the context parameter", ff.Name())
				}
				continue
			}

			// check is thunk or bound
			if strings.HasSuffix(key, "$thunk") || strings.HasSuffix(key, "$bound") {
				continue
			}

			// if ff has no ctx, start deep traversal check
			if !r.checkIsEntry(ff, instr.Pos()) {
				r.buildPkg(ff)

				checkingMap := make(map[string]bool)
				checkingMap[key] = true
				valid := r.checkFuncWithoutCtx(ff, checkingMap)
				setValue(key, valid)
				if !valid {
					r.pass.Reportf(instr.Pos(), "Function `%s` should pass the context parameter", ff.Name())
				}
			}
		}
	}
}

func (r *runner) checkFuncWithoutCtx(f *ssa.Function, checkingMap map[string]bool) (ret bool) {
	ret = true
	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			tp, ok := r.getCtxType(instr)
			if !ok {
				continue
			}

			if tp&CtxOut != 0 {
				continue
			}

			// it is considered illegal as long as ctx is in the input and not in *struct X
			if tp&CtxIn != 0 {
				if tp&CtxInField == 0 {
					ret = false
				}
				continue
			}

			ff := r.getFunction(instr)
			if ff == nil {
				continue
			}

			key := ff.RelString(nil)
			valid, ok := getValue(key)
			if ok {
				if !valid {
					ret = false
					r.pass.Reportf(instr.Pos(), "Function `%s` should pass the context parameter", ff.Name())
				}
				continue
			}

			// check is thunk or bound
			if strings.HasSuffix(key, "$thunk") || strings.HasSuffix(key, "$bound") {
				continue
			}

			if !r.checkIsEntry(ff, instr.Pos()) {
				// handler ring call
				if checkingMap[key] {
					continue
				}
				checkingMap[key] = true

				r.buildPkg(ff)

				valid := r.checkFuncWithoutCtx(ff, checkingMap)
				setValue(key, valid)
				if !valid {
					ret = false
					r.pass.Reportf(instr.Pos(), "Function `%s` should pass the context parameter", ff.Name())
				}
			}
		}
	}
	return ret
}

func (r *runner) getCtxType(instr ssa.Instruction) (tp int, ok bool) {
	switch i := instr.(type) {
	case ssa.CallInstruction:
		tp = r.getCallInstrCtxType(i)
		ok = true
	case *ssa.MakeClosure:
		tp = r.getMakeClosureCtxType(i)
		ok = true
	}
	return
}

func (r *runner) getCallInstrCtxType(c ssa.CallInstruction) (tp int) {
	// check params
	for _, v := range c.Common().Args {
		if r.isCtxType(v.Type()) {
			if vv, ok := v.(*ssa.UnOp); ok {
				if _, ok := vv.X.(*ssa.FieldAddr); ok {
					tp |= CtxInField
				}
			}

			tp |= CtxIn
			break
		}
	}

	// check results
	if v := c.Value(); v != nil {
		if r.isCtxType(v.Type()) {
			tp |= CtxOut
		} else {
			tuple, ok := v.Type().(*types.Tuple)
			if !ok {
				return
			}
			for i := 0; i < tuple.Len(); i++ {
				if r.isCtxType(tuple.At(i).Type()) {
					tp |= CtxOut
					break
				}
			}
		}
	}

	return
}

func (r *runner) getMakeClosureCtxType(c *ssa.MakeClosure) (tp int) {
	for _, v := range c.Bindings {
		if r.isCtxType(v.Type()) {
			if vv, ok := v.(*ssa.UnOp); ok {
				if _, ok := vv.X.(*ssa.FieldAddr); ok {
					tp |= CtxInField
				}
			}

			tp |= CtxIn
			break
		}
	}
	return
}

func (r *runner) getFunction(instr ssa.Instruction) (f *ssa.Function) {
	switch i := instr.(type) {
	case ssa.CallInstruction:
		if i.Common().IsInvoke() {
			return
		}

		switch c := i.Common().Value.(type) {
		case *ssa.Function:
			f = c
		case *ssa.MakeClosure:
			// captured in the outer layer
		case *ssa.Builtin, *ssa.UnOp, *ssa.Lookup, *ssa.Phi:
			// skipped
		case *ssa.Extract, *ssa.Call:
			// function is a result of a call, skipped
		case *ssa.Parameter:
			// function is a param, skipped
		}
	case *ssa.MakeClosure:
		f = i.Fn.(*ssa.Function)
	}
	return
}

func (r *runner) isCtxType(tp types.Type) bool {
	return types.Identical(tp, r.ctxTyp) || types.Identical(tp, r.ctxPTyp)
}

func getValue(key string) (valid, ok bool) {
	checkedMapLock.RLock()
	valid, ok = checkedMap[key]
	checkedMapLock.RUnlock()
	return
}

func setValue(key string, valid bool) {
	checkedMapLock.Lock()
	checkedMap[key] = valid
	checkedMapLock.Unlock()
}
