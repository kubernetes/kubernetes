package nilness

import (
	"fmt"
	"go/token"
	"go/types"
	"reflect"

	"honnef.co/go/tools/go/ir"
	"honnef.co/go/tools/go/types/typeutil"
	"honnef.co/go/tools/internal/passes/buildir"

	"golang.org/x/tools/go/analysis"
)

// neverReturnsNilFact denotes that a function's return value will never
// be nil (typed or untyped). The analysis errs on the side of false
// negatives.
type neverReturnsNilFact struct {
	Rets []neverNilness
}

func (*neverReturnsNilFact) AFact() {}
func (fact *neverReturnsNilFact) String() string {
	return fmt.Sprintf("never returns nil: %v", fact.Rets)
}

type Result struct {
	m map[*types.Func][]neverNilness
}

var Analysis = &analysis.Analyzer{
	Name:       "nilness",
	Doc:        "Annotates return values that will never be nil (typed or untyped)",
	Run:        run,
	Requires:   []*analysis.Analyzer{buildir.Analyzer},
	FactTypes:  []analysis.Fact{(*neverReturnsNilFact)(nil)},
	ResultType: reflect.TypeOf((*Result)(nil)),
}

// MayReturnNil reports whether the ret's return value of fn might be
// a typed or untyped nil value. The value of ret is zero-based. When
// globalOnly is true, the only possible nil values are global
// variables.
//
// The analysis has false positives: MayReturnNil can incorrectly
// report true, but never incorrectly reports false.
func (r *Result) MayReturnNil(fn *types.Func, ret int) (yes bool, globalOnly bool) {
	if !typeutil.IsPointerLike(fn.Type().(*types.Signature).Results().At(ret).Type()) {
		return false, false
	}
	if len(r.m[fn]) == 0 {
		return true, false
	}

	v := r.m[fn][ret]
	return v != neverNil, v == onlyGlobal
}

func run(pass *analysis.Pass) (interface{}, error) {
	seen := map[*ir.Function]struct{}{}
	out := &Result{
		m: map[*types.Func][]neverNilness{},
	}
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		impl(pass, fn, seen)
	}

	for _, fact := range pass.AllObjectFacts() {
		out.m[fact.Object.(*types.Func)] = fact.Fact.(*neverReturnsNilFact).Rets
	}

	return out, nil
}

type neverNilness uint8

const (
	neverNil   neverNilness = 1
	onlyGlobal neverNilness = 2
	nilly      neverNilness = 3
)

func (n neverNilness) String() string {
	switch n {
	case neverNil:
		return "never"
	case onlyGlobal:
		return "global"
	case nilly:
		return "nil"
	default:
		return "BUG"
	}
}

func impl(pass *analysis.Pass, fn *ir.Function, seenFns map[*ir.Function]struct{}) []neverNilness {
	if fn.Object() == nil {
		// TODO(dh): support closures
		return nil
	}
	if fact := new(neverReturnsNilFact); pass.ImportObjectFact(fn.Object(), fact) {
		return fact.Rets
	}
	if fn.Pkg != pass.ResultOf[buildir.Analyzer].(*buildir.IR).Pkg {
		return nil
	}
	if fn.Blocks == nil {
		return nil
	}
	if _, ok := seenFns[fn]; ok {
		// break recursion
		return nil
	}

	seenFns[fn] = struct{}{}

	seen := map[ir.Value]struct{}{}

	var mightReturnNil func(v ir.Value) neverNilness
	mightReturnNil = func(v ir.Value) neverNilness {
		if _, ok := seen[v]; ok {
			// break cycle
			return nilly
		}
		if !typeutil.IsPointerLike(v.Type()) {
			return neverNil
		}
		seen[v] = struct{}{}
		switch v := v.(type) {
		case *ir.MakeInterface:
			return mightReturnNil(v.X)
		case *ir.Convert:
			return mightReturnNil(v.X)
		case *ir.SliceToArrayPointer:
			if v.Type().Underlying().(*types.Pointer).Elem().Underlying().(*types.Array).Len() == 0 {
				return mightReturnNil(v.X)
			} else {
				// converting a slice to an array pointer of length > 0 panics if the slice is nil
				return neverNil
			}
		case *ir.Slice:
			return mightReturnNil(v.X)
		case *ir.Phi:
			ret := neverNil
			for _, e := range v.Edges {
				if n := mightReturnNil(e); n > ret {
					ret = n
				}
			}
			return ret
		case *ir.Extract:
			switch d := v.Tuple.(type) {
			case *ir.Call:
				if callee := d.Call.StaticCallee(); callee != nil {
					ret := impl(pass, callee, seenFns)
					if len(ret) == 0 {
						return nilly
					}
					return ret[v.Index]
				} else {
					return nilly
				}
			case *ir.TypeAssert, *ir.Next, *ir.Select, *ir.MapLookup, *ir.TypeSwitch, *ir.Recv:
				// we don't need to look at the Extract's index
				// because we've already checked its type.
				return nilly
			default:
				panic(fmt.Sprintf("internal error: unhandled type %T", d))
			}
		case *ir.Call:
			if callee := v.Call.StaticCallee(); callee != nil {
				ret := impl(pass, callee, seenFns)
				if len(ret) == 0 {
					return nilly
				}
				return ret[0]
			} else {
				return nilly
			}
		case *ir.BinOp, *ir.UnOp, *ir.Alloc, *ir.FieldAddr, *ir.IndexAddr, *ir.Global, *ir.MakeSlice, *ir.MakeClosure, *ir.Function, *ir.MakeMap, *ir.MakeChan:
			return neverNil
		case *ir.Sigma:
			iff, ok := v.From.Control().(*ir.If)
			if !ok {
				return nilly
			}
			binop, ok := iff.Cond.(*ir.BinOp)
			if !ok {
				return nilly
			}
			isNil := func(v ir.Value) bool {
				k, ok := v.(*ir.Const)
				if !ok {
					return false
				}
				return k.Value == nil
			}
			if binop.X == v.X && isNil(binop.Y) || binop.Y == v.X && isNil(binop.X) {
				op := binop.Op
				if v.From.Succs[0] != v.Block() {
					// we're in the false branch, negate op
					switch op {
					case token.EQL:
						op = token.NEQ
					case token.NEQ:
						op = token.EQL
					default:
						panic(fmt.Sprintf("internal error: unhandled token %v", op))
					}
				}
				switch op {
				case token.EQL:
					return nilly
				case token.NEQ:
					return neverNil
				default:
					panic(fmt.Sprintf("internal error: unhandled token %v", op))
				}
			}
			return nilly
		case *ir.ChangeType:
			return mightReturnNil(v.X)
		case *ir.Load:
			if _, ok := v.X.(*ir.Global); ok {
				return onlyGlobal
			}
			return nilly
		case *ir.TypeAssert, *ir.ChangeInterface, *ir.Field, *ir.Const, *ir.Index, *ir.MapLookup, *ir.Parameter, *ir.Recv, *ir.TypeSwitch:
			return nilly
		default:
			panic(fmt.Sprintf("internal error: unhandled type %T", v))
		}
	}
	ret := fn.Exit.Control().(*ir.Return)
	out := make([]neverNilness, len(ret.Results))
	export := false
	for i, v := range ret.Results {
		v := mightReturnNil(v)
		out[i] = v
		if v != nilly && typeutil.IsPointerLike(fn.Signature.Results().At(i).Type()) {
			export = true
		}
	}
	if export {
		pass.ExportObjectFact(fn.Object(), &neverReturnsNilFact{out})
	}
	return out
}
