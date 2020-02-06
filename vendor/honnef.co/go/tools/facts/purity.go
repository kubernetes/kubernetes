package facts

import (
	"go/token"
	"go/types"
	"reflect"

	"golang.org/x/tools/go/analysis"
	"honnef.co/go/tools/functions"
	"honnef.co/go/tools/internal/passes/buildssa"
	"honnef.co/go/tools/ssa"
)

type IsPure struct{}

func (*IsPure) AFact()           {}
func (d *IsPure) String() string { return "is pure" }

type PurityResult map[*types.Func]*IsPure

var Purity = &analysis.Analyzer{
	Name:       "fact_purity",
	Doc:        "Mark pure functions",
	Run:        purity,
	Requires:   []*analysis.Analyzer{buildssa.Analyzer},
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
	seen := map[*ssa.Function]struct{}{}
	ssapkg := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA).Pkg
	var check func(ssafn *ssa.Function) (ret bool)
	check = func(ssafn *ssa.Function) (ret bool) {
		if ssafn.Object() == nil {
			// TODO(dh): support closures
			return false
		}
		if pass.ImportObjectFact(ssafn.Object(), new(IsPure)) {
			return true
		}
		if ssafn.Pkg != ssapkg {
			// Function is in another package but wasn't marked as
			// pure, ergo it isn't pure
			return false
		}
		// Break recursion
		if _, ok := seen[ssafn]; ok {
			return false
		}

		seen[ssafn] = struct{}{}
		defer func() {
			if ret {
				pass.ExportObjectFact(ssafn.Object(), &IsPure{})
			}
		}()

		if functions.IsStub(ssafn) {
			return false
		}

		if _, ok := pureStdlib[ssafn.Object().(*types.Func).FullName()]; ok {
			return true
		}

		if ssafn.Signature.Results().Len() == 0 {
			// A function with no return values is empty or is doing some
			// work we cannot see (for example because of build tags);
			// don't consider it pure.
			return false
		}

		for _, param := range ssafn.Params {
			if _, ok := param.Type().Underlying().(*types.Basic); !ok {
				return false
			}
		}

		if ssafn.Blocks == nil {
			return false
		}
		checkCall := func(common *ssa.CallCommon) bool {
			if common.IsInvoke() {
				return false
			}
			builtin, ok := common.Value.(*ssa.Builtin)
			if !ok {
				if common.StaticCallee() != ssafn {
					if common.StaticCallee() == nil {
						return false
					}
					if !check(common.StaticCallee()) {
						return false
					}
				}
			} else {
				switch builtin.Name() {
				case "len", "cap", "make", "new":
				default:
					return false
				}
			}
			return true
		}
		for _, b := range ssafn.Blocks {
			for _, ins := range b.Instrs {
				switch ins := ins.(type) {
				case *ssa.Call:
					if !checkCall(ins.Common()) {
						return false
					}
				case *ssa.Defer:
					if !checkCall(&ins.Call) {
						return false
					}
				case *ssa.Select:
					return false
				case *ssa.Send:
					return false
				case *ssa.Go:
					return false
				case *ssa.Panic:
					return false
				case *ssa.Store:
					return false
				case *ssa.FieldAddr:
					return false
				case *ssa.UnOp:
					if ins.Op == token.MUL || ins.Op == token.AND {
						return false
					}
				}
			}
		}
		return true
	}
	for _, ssafn := range pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA).SrcFuncs {
		check(ssafn)
	}

	out := PurityResult{}
	for _, fact := range pass.AllObjectFacts() {
		out[fact.Object.(*types.Func)] = fact.Fact.(*IsPure)
	}
	return out, nil
}
