package typedness

import (
	"fmt"
	"go/token"
	"go/types"
	"reflect"

	"honnef.co/go/tools/go/ir"
	"honnef.co/go/tools/go/ir/irutil"
	"honnef.co/go/tools/internal/passes/buildir"

	"golang.org/x/tools/go/analysis"
)

// alwaysTypedFact denotes that a function's return value will never
// be untyped nil. The analysis errs on the side of false negatives.
type alwaysTypedFact struct {
	Rets uint8
}

func (*alwaysTypedFact) AFact() {}
func (fact *alwaysTypedFact) String() string {
	return fmt.Sprintf("always typed: %08b", fact.Rets)
}

type Result struct {
	m map[*types.Func]uint8
}

var Analysis = &analysis.Analyzer{
	Name:       "typedness",
	Doc:        "Annotates return values that are always typed values",
	Run:        run,
	Requires:   []*analysis.Analyzer{buildir.Analyzer},
	FactTypes:  []analysis.Fact{(*alwaysTypedFact)(nil)},
	ResultType: reflect.TypeOf((*Result)(nil)),
}

// MustReturnTyped reports whether the ret's return value of fn must
// be a typed value, i.e. an interface value containing a concrete
// type or trivially a concrete type. The value of ret is zero-based.
//
// The analysis has false negatives: MustReturnTyped may incorrectly
// report false, but never incorrectly reports true.
func (r *Result) MustReturnTyped(fn *types.Func, ret int) bool {
	if _, ok := fn.Type().(*types.Signature).Results().At(ret).Type().Underlying().(*types.Interface); !ok {
		return true
	}
	return (r.m[fn] & (1 << ret)) != 0
}

func run(pass *analysis.Pass) (interface{}, error) {
	seen := map[*ir.Function]struct{}{}
	out := &Result{
		m: map[*types.Func]uint8{},
	}
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		impl(pass, fn, seen)
	}

	for _, fact := range pass.AllObjectFacts() {
		out.m[fact.Object.(*types.Func)] = fact.Fact.(*alwaysTypedFact).Rets
	}

	return out, nil
}

func impl(pass *analysis.Pass, fn *ir.Function, seenFns map[*ir.Function]struct{}) (out uint8) {
	if fn.Signature.Results().Len() > 8 {
		return 0
	}
	if fn.Object() == nil {
		// TODO(dh): support closures
		return 0
	}
	if fact := new(alwaysTypedFact); pass.ImportObjectFact(fn.Object(), fact) {
		return fact.Rets
	}
	if fn.Pkg != pass.ResultOf[buildir.Analyzer].(*buildir.IR).Pkg {
		return 0
	}
	if fn.Blocks == nil {
		return 0
	}
	if irutil.IsStub(fn) {
		return 0
	}
	if _, ok := seenFns[fn]; ok {
		// break recursion
		return 0
	}

	seenFns[fn] = struct{}{}
	defer func() {
		for i := 0; i < fn.Signature.Results().Len(); i++ {
			if _, ok := fn.Signature.Results().At(i).Type().Underlying().(*types.Interface); !ok {
				// we don't need facts to know that non-interface
				// types can't be untyped nil. zeroing out those bits
				// may result in all bits being zero, in which case we
				// don't have to save any fact.
				out &= ^(1 << i)
			}
		}
		if out > 0 {
			pass.ExportObjectFact(fn.Object(), &alwaysTypedFact{out})
		}
	}()

	isUntypedNil := func(v ir.Value) bool {
		k, ok := v.(*ir.Const)
		if !ok {
			return false
		}
		if _, ok := k.Type().Underlying().(*types.Interface); !ok {
			return false
		}
		return k.Value == nil
	}

	var do func(v ir.Value, seen map[ir.Value]struct{}) bool
	do = func(v ir.Value, seen map[ir.Value]struct{}) bool {
		if _, ok := seen[v]; ok {
			// break cycle
			return false
		}
		seen[v] = struct{}{}
		switch v := v.(type) {
		case *ir.Const:
			// can't be a typed nil, because then we'd be returning the
			// result of MakeInterface.
			return false
		case *ir.ChangeInterface:
			return do(v.X, seen)
		case *ir.Extract:
			call, ok := v.Tuple.(*ir.Call)
			if !ok {
				// We only care about extracts of function results. For
				// everything else (e.g. channel receives and map
				// lookups), we can either not deduce any information, or
				// will see a MakeInterface.
				return false
			}
			if callee := call.Call.StaticCallee(); callee != nil {
				return impl(pass, callee, seenFns)&(1<<v.Index) != 0
			} else {
				// we don't know what function we're calling. no need
				// to look at the signature, though. if it weren't an
				// interface, we'd be seeing a MakeInterface
				// instruction.
				return false
			}
		case *ir.Call:
			if callee := v.Call.StaticCallee(); callee != nil {
				return impl(pass, callee, seenFns)&1 != 0
			} else {
				// we don't know what function we're calling. no need
				// to look at the signature, though. if it weren't an
				// interface, we'd be seeing a MakeInterface
				// instruction.
				return false
			}
		case *ir.Sigma:
			iff, ok := v.From.Control().(*ir.If)
			if !ok {
				// give up
				return false
			}

			binop, ok := iff.Cond.(*ir.BinOp)
			if !ok {
				// give up
				return false
			}

			if (binop.X == v.X && isUntypedNil(binop.Y)) || (isUntypedNil(binop.X) && binop.Y == v.X) {
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
					// returned value equals untyped nil
					return false
				case token.NEQ:
					// returned value does not equal untyped nil
					return true
				default:
					panic(fmt.Sprintf("internal error: unhandled token %v", op))
				}
			}

			// TODO(dh): handle comparison with typed nil

			// give up
			return false
		case *ir.Phi:
			for _, pv := range v.Edges {
				if !do(pv, seen) {
					return false
				}
			}
			return true
		case *ir.MakeInterface:
			return true
		case *ir.TypeAssert:
			// type assertions fail for untyped nils. Either we have a
			// single lhs and the type assertion succeeds or panics,
			// or we have two lhs and we'll return Extract instead.
			return true
		case *ir.ChangeType:
			// we'll only see interface->interface conversions, which
			// don't tell us anything about the nilness.
			return false
		case *ir.MapLookup, *ir.Index, *ir.Recv, *ir.Parameter, *ir.Load, *ir.Field:
			// All other instructions that tell us nothing about the
			// typedness of interface values.
			return false
		default:
			panic(fmt.Sprintf("internal error: unhandled type %T", v))
		}
	}

	ret := fn.Exit.Control().(*ir.Return)
	for i, v := range ret.Results {
		if _, ok := fn.Signature.Results().At(i).Type().Underlying().(*types.Interface); ok {
			if do(v, map[ir.Value]struct{}{}) {
				out |= 1 << i
			}
		}
	}
	return out
}
