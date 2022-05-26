package nilerr

import (
	"fmt"
	"go/token"
	"go/types"

	"github.com/gostaticanalysis/comment"
	"github.com/gostaticanalysis/comment/passes/commentmap"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"
	"golang.org/x/tools/go/ssa"
)

var Analyzer = &analysis.Analyzer{
	Name: "nilerr",
	Doc:  Doc,
	Run:  run,
	Requires: []*analysis.Analyzer{
		buildssa.Analyzer,
		commentmap.Analyzer,
	},
}

const Doc = "nilerr checks returning nil when err is not nil"

func run(pass *analysis.Pass) (interface{}, error) {
	funcs := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA).SrcFuncs
	cmaps := pass.ResultOf[commentmap.Analyzer].(comment.Maps)

	reportFail := func(v ssa.Value, ret *ssa.Return, format string) {
		pos := ret.Pos()
		line := getNodeLineNumber(pass, ret)
		errLines := getValueLineNumbers(pass, v)
		if !cmaps.IgnoreLine(pass.Fset, line, "nilerr") {
			var errLineText string
			if len(errLines) == 1 {
				errLineText = fmt.Sprintf("line %d", errLines[0])
			} else {
				errLineText = fmt.Sprintf("lines %v", errLines)
			}
			pass.Reportf(pos, format, errLineText)
		}
	}

	for i := range funcs {
		for _, b := range funcs[i].Blocks {
			if v := binOpErrNil(b, token.NEQ); v != nil {
				if ret := isReturnNil(b.Succs[0]); ret != nil {
					if !usesErrorValue(b.Succs[0], v) {
						reportFail(v, ret, "error is not nil (%s) but it returns nil")
					}
				}
			} else if v := binOpErrNil(b, token.EQL); v != nil {
				if len(b.Succs[0].Preds) == 1 { // if there are multiple conditions, this may be false positive
					if ret := isReturnError(b.Succs[0], v); ret != nil {
						reportFail(v, ret, "error is nil (%s) but it returns error")
					}
				}
			}

		}
	}

	return nil, nil
}

func getValueLineNumbers(pass *analysis.Pass, v ssa.Value) []int {
	if phi, ok := v.(*ssa.Phi); ok {
		result := make([]int, 0, len(phi.Edges))
		for _, edge := range phi.Edges {
			result = append(result, getValueLineNumbers(pass, edge)...)
		}
		return result
	}

	value := v
	if extract, ok := value.(*ssa.Extract); ok {
		value = extract.Tuple
	}

	pos := value.Pos()
	return []int{pass.Fset.File(pos).Line(pos)}
}

func getNodeLineNumber(pass *analysis.Pass, node ssa.Node) int {
	pos := node.Pos()
	return pass.Fset.File(pos).Line(pos)
}

var errType = types.Universe.Lookup("error").Type().Underlying().(*types.Interface)

func binOpErrNil(b *ssa.BasicBlock, op token.Token) ssa.Value {
	if len(b.Instrs) == 0 {
		return nil
	}

	ifinst, ok := b.Instrs[len(b.Instrs)-1].(*ssa.If)
	if !ok {
		return nil
	}

	binop, ok := ifinst.Cond.(*ssa.BinOp)
	if !ok {
		return nil
	}

	if binop.Op != op {
		return nil
	}

	if !types.Implements(binop.X.Type(), errType) {
		return nil
	}

	if !types.Implements(binop.Y.Type(), errType) {
		return nil
	}

	xIsConst, yIsConst := isConst(binop.X), isConst(binop.Y)
	switch {
	case !xIsConst && yIsConst: // err != nil or err == nil
		return binop.X
	case xIsConst && !yIsConst: // nil != err or nil == err
		return binop.Y
	}

	return nil
}

func isConst(v ssa.Value) bool {
	_, ok := v.(*ssa.Const)
	return ok
}

func isReturnNil(b *ssa.BasicBlock) *ssa.Return {
	if len(b.Instrs) == 0 {
		return nil
	}

	ret, ok := b.Instrs[len(b.Instrs)-1].(*ssa.Return)
	if !ok {
		return nil
	}

	errorReturnValues := 0
	for _, res := range ret.Results {
		if !types.Implements(res.Type(), errType) {
			continue
		}

		errorReturnValues++
		v, ok := res.(*ssa.Const)
		if !ok {
			return nil
		}

		if !v.IsNil() {
			return nil
		}
	}

	if errorReturnValues == 0 {
		return nil
	}

	return ret
}

func isReturnError(b *ssa.BasicBlock, errVal ssa.Value) *ssa.Return {
	if len(b.Instrs) == 0 {
		return nil
	}

	ret, ok := b.Instrs[len(b.Instrs)-1].(*ssa.Return)
	if !ok {
		return nil
	}

	for _, v := range ret.Results {
		if v == errVal {
			return ret
		}
	}

	return nil
}

func usesErrorValue(b *ssa.BasicBlock, errVal ssa.Value) bool {
	for _, instr := range b.Instrs {
		if callInstr, ok := instr.(*ssa.Call); ok {
			for _, arg := range callInstr.Call.Args {
				if isUsedInValue(arg, errVal) {
					return true
				}

				sliceArg, ok := arg.(*ssa.Slice)
				if ok {
					if isUsedInSlice(sliceArg, errVal) {
						return true
					}
				}
			}
		}
	}
	return false
}

type ReferrersHolder interface {
	Referrers() *[]ssa.Instruction
}

var _ ReferrersHolder = (ssa.Node)(nil)
var _ ReferrersHolder = (ssa.Value)(nil)

func isUsedInSlice(sliceArg *ssa.Slice, errVal ssa.Value) bool {
	var valueBuf [10]*ssa.Value
	operands := sliceArg.Operands(valueBuf[:0])

	var valuesToInspect []ssa.Value
	addValueForInspection := func(value ssa.Value) {
		if value != nil {
			valuesToInspect = append(valuesToInspect, value)
		}
	}

	var nodesToInspect []ssa.Node
	visitedNodes := map[ssa.Node]bool{}
	addNodeForInspection := func(node ssa.Node) {
		if !visitedNodes[node] {
			visitedNodes[node] = true
			nodesToInspect = append(nodesToInspect, node)
		}
	}
	addReferrersForInspection := func(h ReferrersHolder) {
		if h == nil {
			return
		}

		referrers := h.Referrers()
		if referrers == nil {
			return
		}

		for _, r := range *referrers {
			if node, ok := r.(ssa.Node); ok {
				addNodeForInspection(node)
			}
		}
	}

	for _, operand := range operands {
		addReferrersForInspection(*operand)
		addValueForInspection(*operand)
	}

	for i := 0; i < len(nodesToInspect); i++ {
		switch node := nodesToInspect[i].(type) {
		case *ssa.IndexAddr:
			addReferrersForInspection(node)
		case *ssa.Store:
			addValueForInspection(node.Val)
		}
	}

	for _, value := range valuesToInspect {
		if isUsedInValue(value, errVal) {
			return true
		}
	}
	return false
}

func isUsedInValue(value, lookedFor ssa.Value) bool {
	if value == lookedFor {
		return true
	}

	switch value := value.(type) {
	case *ssa.ChangeInterface:
		return isUsedInValue(value.X, lookedFor)
	case *ssa.MakeInterface:
		return isUsedInValue(value.X, lookedFor)
	case *ssa.Call:
		if value.Call.IsInvoke() {
			return isUsedInValue(value.Call.Value, lookedFor)
		}
	}

	return false
}
