package reqwithoutctx

import (
	"go/types"

	"github.com/gostaticanalysis/analysisutil"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"
	"golang.org/x/tools/go/ssa"
)

type Analyzer struct {
	Funcs          []*ssa.Function
	newRequestType types.Type
	requestType    types.Type
}

func NewAnalyzer(pass *analysis.Pass) *Analyzer {
	newRequestType := analysisutil.TypeOf(pass, "net/http", "NewRequest")
	requestType := analysisutil.TypeOf(pass, "net/http", "*Request")

	srcFuncs := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA).SrcFuncs

	return &Analyzer{
		Funcs:          srcFuncs,
		newRequestType: newRequestType,
		requestType:    requestType,
	}
}

func (a *Analyzer) Exec() []*Report {
	if a.newRequestType == nil || a.requestType == nil {
		return []*Report{}
	}

	usedReqs := a.usedReqs()
	newReqs := a.requestsByNewRequest()

	return a.report(usedReqs, newReqs)
}

func (a *Analyzer) report(usedReqs map[string]*ssa.Extract, newReqs map[*ssa.Call]*ssa.Extract) []*Report {
	var reports []*Report

	for _, fReq := range usedReqs {
		for newRequest, req := range newReqs {
			if fReq == req {
				reports = append(reports, &Report{Instruction: newRequest})
			}
		}
	}

	return reports
}

func (a *Analyzer) usedReqs() map[string]*ssa.Extract {
	reqExts := make(map[string]*ssa.Extract)

	for _, f := range a.Funcs {
		for _, b := range f.Blocks {
			for _, instr := range b.Instrs {
				switch i := instr.(type) {
				case *ssa.Call:
					exts := a.usedReqByCall(i)
					for _, ext := range exts {
						key := i.String() + ext.String()
						reqExts[key] = ext
					}
				case *ssa.UnOp:
					ext := a.usedReqByUnOp(i)
					if ext != nil {
						key := i.String() + ext.String()
						reqExts[key] = ext
					}
				case *ssa.Return:
					exts := a.usedReqByReturn(i)
					for _, ext := range exts {
						key := i.String() + ext.String()
						reqExts[key] = ext
					}
				}
			}
		}
	}

	return reqExts
}

func (a *Analyzer) usedReqByCall(call *ssa.Call) []*ssa.Extract {
	var exts []*ssa.Extract

	// skip net/http.Request method call
	if call.Common().Signature().Recv() != nil && types.Identical(call.Value().Type(), a.requestType) {
		return exts
	}

	args := call.Common().Args
	if len(args) == 0 {
		return exts
	}

	for _, arg := range args {
		ext, ok := arg.(*ssa.Extract)
		if !ok {
			continue
		}

		if !types.Identical(ext.Type(), a.requestType) {
			continue
		}

		exts = append(exts, ext)
	}

	return exts
}

func (a *Analyzer) usedReqByUnOp(op *ssa.UnOp) *ssa.Extract {
	if ext, ok := op.X.(*ssa.Extract); ok && types.Identical(ext.Type(), a.requestType) {
		return ext
	}

	return nil
}

func (a *Analyzer) usedReqByReturn(ret *ssa.Return) []*ssa.Extract {
	rets := ret.Results
	exts := make([]*ssa.Extract, 0, len(rets))

	for _, ret := range rets {
		ext, ok := ret.(*ssa.Extract)
		if !ok {
			continue
		}

		if types.Identical(ext.Type(), a.requestType) {
			exts = append(exts, ext)
		}
	}

	return exts
}

func (a *Analyzer) requestsByNewRequest() map[*ssa.Call]*ssa.Extract {
	reqs := make(map[*ssa.Call]*ssa.Extract)

	for _, f := range a.Funcs {
		for _, b := range f.Blocks {
			for _, instr := range b.Instrs {
				ext, ok := instr.(*ssa.Extract)
				if !ok {
					continue
				}

				if !types.Identical(ext.Type(), a.requestType) {
					continue
				}

				operands := ext.Operands([]*ssa.Value{})
				if len(operands) != 1 {
					continue
				}

				operand := *operands[0]

				f, ok := operand.(*ssa.Call)
				if !ok {
					continue
				}

				if types.Identical(f.Call.Value.Type(), a.newRequestType) {
					reqs[f] = ext
				}
			}
		}
	}

	return reqs
}
