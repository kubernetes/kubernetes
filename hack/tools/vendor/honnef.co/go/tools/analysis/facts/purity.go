package facts

import (
	"go/types"
	"reflect"

	"honnef.co/go/tools/go/ir"
	"honnef.co/go/tools/go/ir/irutil"
	"honnef.co/go/tools/internal/passes/buildir"

	"golang.org/x/tools/go/analysis"
)

type IsPure struct{}

func (*IsPure) AFact()           {}
func (d *IsPure) String() string { return "is pure" }

type PurityResult map[*types.Func]*IsPure

var Purity = &analysis.Analyzer{
	Name:       "fact_purity",
	Doc:        "Mark pure functions",
	Run:        purity,
	Requires:   []*analysis.Analyzer{buildir.Analyzer},
	FactTypes:  []analysis.Fact{(*IsPure)(nil)},
	ResultType: reflect.TypeOf(PurityResult{}),
}

var pureStdlib = map[string]struct{}{
	"errors.New":                      {},
	"fmt.Errorf":                      {},
	"fmt.Sprintf":                     {},
	"fmt.Sprint":                      {},
	"sort.Reverse":                    {},
	"strings.Map":                     {},
	"strings.Repeat":                  {},
	"strings.Replace":                 {},
	"strings.Title":                   {},
	"strings.ToLower":                 {},
	"strings.ToLowerSpecial":          {},
	"strings.ToTitle":                 {},
	"strings.ToTitleSpecial":          {},
	"strings.ToUpper":                 {},
	"strings.ToUpperSpecial":          {},
	"strings.Trim":                    {},
	"strings.TrimFunc":                {},
	"strings.TrimLeft":                {},
	"strings.TrimLeftFunc":            {},
	"strings.TrimPrefix":              {},
	"strings.TrimRight":               {},
	"strings.TrimRightFunc":           {},
	"strings.TrimSpace":               {},
	"strings.TrimSuffix":              {},
	"(*net/http.Request).WithContext": {},
}

func purity(pass *analysis.Pass) (interface{}, error) {
	seen := map[*ir.Function]struct{}{}
	irpkg := pass.ResultOf[buildir.Analyzer].(*buildir.IR).Pkg
	var check func(fn *ir.Function) (ret bool)
	check = func(fn *ir.Function) (ret bool) {
		if fn.Object() == nil {
			// TODO(dh): support closures
			return false
		}
		if pass.ImportObjectFact(fn.Object(), new(IsPure)) {
			return true
		}
		if fn.Pkg != irpkg {
			// Function is in another package but wasn't marked as
			// pure, ergo it isn't pure
			return false
		}
		// Break recursion
		if _, ok := seen[fn]; ok {
			return false
		}

		seen[fn] = struct{}{}
		defer func() {
			if ret {
				pass.ExportObjectFact(fn.Object(), &IsPure{})
			}
		}()

		if irutil.IsStub(fn) {
			return false
		}

		if _, ok := pureStdlib[fn.Object().(*types.Func).FullName()]; ok {
			return true
		}

		if fn.Signature.Results().Len() == 0 {
			// A function with no return values is empty or is doing some
			// work we cannot see (for example because of build tags);
			// don't consider it pure.
			return false
		}

		for _, param := range fn.Params {
			// TODO(dh): this may not be strictly correct. pure code
			// can, to an extent, operate on non-basic types.
			if _, ok := param.Type().Underlying().(*types.Basic); !ok {
				return false
			}
		}

		// Don't consider external functions pure.
		if fn.Blocks == nil {
			return false
		}
		checkCall := func(common *ir.CallCommon) bool {
			if common.IsInvoke() {
				return false
			}
			builtin, ok := common.Value.(*ir.Builtin)
			if !ok {
				if common.StaticCallee() != fn {
					if common.StaticCallee() == nil {
						return false
					}
					if !check(common.StaticCallee()) {
						return false
					}
				}
			} else {
				switch builtin.Name() {
				case "len", "cap":
				default:
					return false
				}
			}
			return true
		}
		for _, b := range fn.Blocks {
			for _, ins := range b.Instrs {
				switch ins := ins.(type) {
				case *ir.Call:
					if !checkCall(ins.Common()) {
						return false
					}
				case *ir.Defer:
					if !checkCall(&ins.Call) {
						return false
					}
				case *ir.Select:
					return false
				case *ir.Send:
					return false
				case *ir.Go:
					return false
				case *ir.Panic:
					return false
				case *ir.Store:
					return false
				case *ir.FieldAddr:
					return false
				case *ir.Alloc:
					return false
				case *ir.Load:
					return false
				}
			}
		}
		return true
	}
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		check(fn)
	}

	out := PurityResult{}
	for _, fact := range pass.AllObjectFacts() {
		out[fact.Object.(*types.Func)] = fact.Fact.(*IsPure)
	}
	return out, nil
}
