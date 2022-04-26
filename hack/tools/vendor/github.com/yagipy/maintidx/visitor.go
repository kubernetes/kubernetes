package maintidx

import (
	"github.com/yagipy/maintidx/pkg/cyc"
	"github.com/yagipy/maintidx/pkg/halstvol"
	"go/ast"
	"math"
	"sort"
)

type Visitor struct {
	MaintIdx int
	Coef     Coef
}

var _ ast.Visitor = &Visitor{}

type Coef struct {
	Cyc      cyc.Cyc
	HalstVol halstvol.HalstVol
}

func NewVisitor() *Visitor {
	return &Visitor{
		MaintIdx: 0,
		Coef: Coef{
			Cyc: cyc.Cyc{
				Val:  1,
				Coef: cyc.Coef{},
			},
			HalstVol: halstvol.HalstVol{
				Val: 0.0,
				Coef: halstvol.Coef{
					Opt: map[string]int{},
					Opd: map[string]int{},
				},
			},
		},
	}
}

func (v *Visitor) Visit(n ast.Node) ast.Visitor {
	v.Coef.Cyc.Analyze(n)
	v.Coef.HalstVol.Analyze(n)
	return v
}

// Calc https://docs.microsoft.com/ja-jp/archive/blogs/codeanalysis/maintainability-index-range-and-meaning
func (v *Visitor) calc(loc int) {
	origVal := 171.0 - 5.2*math.Log(v.Coef.HalstVol.Val) - 0.23*float64(v.Coef.Cyc.Val) - 16.2*math.Log(float64(loc))
	normVal := int(math.Max(0.0, origVal*100.0/171.0))
	v.MaintIdx = normVal
}

// TODO: Move halstvol package
func (v *Visitor) printHalstVol() {
	sortedOpt := make([]string, len(v.Coef.HalstVol.Coef.Opt))
	sortedOpd := make([]string, len(v.Coef.HalstVol.Coef.Opd))
	optIndex := 0
	opdIndex := 0
	for key := range v.Coef.HalstVol.Coef.Opt {
		sortedOpt[optIndex] = key
		optIndex++
	}
	for key := range v.Coef.HalstVol.Coef.Opd {
		sortedOpd[opdIndex] = key
		opdIndex++
	}
	sort.Strings(sortedOpt)
	sort.Strings(sortedOpd)
	for _, val := range sortedOpt {
		println("operators", val, v.Coef.HalstVol.Coef.Opt[val])
	}
	for _, val := range sortedOpd {
		println("operands", val, v.Coef.HalstVol.Coef.Opd[val])
	}
}
