package analysisutil

import (
	"go/types"

	"golang.org/x/tools/go/ssa"
)

// CalledChecker checks a function is called.
// See From and Func.
type CalledChecker struct {
	Ignore func(instr ssa.Instruction) bool
}

// NotIn checks whether receiver's method is called in a function.
// If there is no methods calling at a path from an instruction
// which type is receiver to all return instruction, NotIn returns these instructions.
func (c *CalledChecker) NotIn(f *ssa.Function, receiver types.Type, methods ...*types.Func) []ssa.Instruction {
	done := map[ssa.Value]bool{}
	var instrs []ssa.Instruction
	for _, b := range f.Blocks {
		for i, instr := range b.Instrs {
			v, _ := instr.(ssa.Value)
			if v == nil || done[v] {
				continue
			}

			if v, _ := v.(*ssa.UnOp); v != nil && done[v.X] {
				continue
			}

			called, ok := c.From(b, i, receiver, methods...)
			if ok && !called {
				instrs = append(instrs, instr)
				done[v] = true
				if v, _ := v.(*ssa.UnOp); v != nil {
					done[v.X] = true
				}
			}
		}
	}
	return instrs
}

// Func returns true when f is called in the instr.
// If recv is not nil, Func also checks the receiver.
func (c *CalledChecker) Func(instr ssa.Instruction, recv ssa.Value, f *types.Func) bool {

	if c.Ignore != nil && c.Ignore(instr) {
		return false
	}

	call, ok := instr.(ssa.CallInstruction)
	if !ok {
		return false
	}

	common := call.Common()
	if common == nil {
		return false
	}

	callee := common.StaticCallee()
	if callee == nil {
		return false
	}

	fn, ok := callee.Object().(*types.Func)
	if !ok {
		return false
	}

	if recv != nil &&
		common.Signature().Recv() != nil &&
		(len(common.Args) == 0 && recv != nil || common.Args[0] != recv &&
			!referrer(recv, common.Args[0])) {
		return false
	}

	return fn == f
}

func referrer(a, b ssa.Value) bool {
	return isReferrerOf(a, b) || isReferrerOf(b, a)
}

func isReferrerOf(a, b ssa.Value) bool {
	if a == nil || b == nil {
		return false
	}
	if b.Referrers() != nil {
		brs := *b.Referrers()

		for _, br := range brs {
			brv, ok := br.(ssa.Value)
			if !ok {
				continue
			}
			if brv == a {
				return true
			}
		}
	}
	return false
}

// From checks whether receiver's method is called in an instruction
// which belogns to after i-th instructions, or in succsor blocks of b.
// The first result is above value.
// The second result is whether type of i-th instruction does not much receiver
// or matches with ignore cases.
func (c *CalledChecker) From(b *ssa.BasicBlock, i int, receiver types.Type, methods ...*types.Func) (called, ok bool) {
	if b == nil || i < 0 || i >= len(b.Instrs) ||
		receiver == nil || len(methods) == 0 {
		return false, false
	}

	v, ok := b.Instrs[i].(ssa.Value)
	if !ok {
		return false, false
	}

	from := &calledFrom{recv: v, fs: methods, ignore: c.Ignore}

	if !from.isRecv(receiver, v.Type()) {
		return false, false
	}

	if from.ignored() {
		return false, false
	}

	if from.instrs(b.Instrs[i+1:]) ||
		from.succs(b) {
		return true, true
	}

	from.done = nil
	if from.storedInInstrs(b.Instrs[i+1:]) ||
		from.storedInSuccs(b) {
		return false, false
	}

	return false, true
}

type calledFrom struct {
	recv   ssa.Value
	fs     []*types.Func
	done   map[*ssa.BasicBlock]bool
	ignore func(ssa.Instruction) bool
}

func (c *calledFrom) ignored() bool {

	switch v := c.recv.(type) {
	case *ssa.UnOp:
		switch v.X.(type) {
		case *ssa.FreeVar, *ssa.Global:
			return true
		}
	}

	refs := c.recv.Referrers()
	if refs == nil {
		return false
	}

	for _, ref := range *refs {
		done := map[ssa.Instruction]bool{}
		if !c.isOwn(ref) &&
			((c.ignore != nil && c.ignore(ref)) ||
				c.isRet(ref, done) || c.isArg(ref)) {
			return true
		}
	}

	return false
}

func (c *calledFrom) isOwn(instr ssa.Instruction) bool {
	v, ok := instr.(ssa.Value)
	if !ok {
		return false
	}
	return v == c.recv
}

func (c *calledFrom) isRet(instr ssa.Instruction, done map[ssa.Instruction]bool) bool {
	if done[instr] {
		return false
	}
	done[instr] = true

	switch instr := instr.(type) {
	case *ssa.Return:
		return true
	case *ssa.MapUpdate:
		return c.isRetInRefs(instr.Map, done)
	case *ssa.Store:
		if instr, _ := instr.Addr.(ssa.Instruction); instr != nil {
			return c.isRet(instr, done)
		}
		return c.isRetInRefs(instr.Addr, done)
	case *ssa.FieldAddr:
		return c.isRetInRefs(instr.X, done)
	case ssa.Value:
		return c.isRetInRefs(instr, done)
	default:
		return false
	}
}

func (c *calledFrom) isRetInRefs(v ssa.Value, done map[ssa.Instruction]bool) bool {
	refs := v.Referrers()
	if refs == nil {
		return false
	}
	for _, ref := range *refs {
		if c.isRet(ref, done) {
			return true
		}
	}
	return false
}

func (c *calledFrom) isArg(instr ssa.Instruction) bool {

	call, ok := instr.(ssa.CallInstruction)
	if !ok {
		return false
	}

	common := call.Common()
	if common == nil {
		return false
	}

	args := common.Args
	if common.Signature().Recv() != nil {
		args = args[1:]
	}

	for i := range args {
		if args[i] == c.recv {
			return true
		}
	}

	return false
}

func (c *calledFrom) instrs(instrs []ssa.Instruction) bool {
	for _, instr := range instrs {
		for _, f := range c.fs {
			if Called(instr, c.recv, f) {
				return true
			}
		}
	}
	return false
}

func (c *calledFrom) succs(b *ssa.BasicBlock) bool {
	if c.done == nil {
		c.done = map[*ssa.BasicBlock]bool{}
	}

	if c.done[b] {
		return true
	}
	c.done[b] = true

	if len(b.Succs) == 0 {
		return false
	}

	for _, s := range b.Succs {
		if !c.instrs(s.Instrs) && !c.succs(s) {
			return false
		}
	}

	return true
}

func (c *calledFrom) storedInInstrs(instrs []ssa.Instruction) bool {
	for _, instr := range instrs {
		switch instr := instr.(type) {
		case *ssa.Store:
			if instr.Val == c.recv {
				return true
			}
		}
	}
	return false
}

func (c *calledFrom) storedInSuccs(b *ssa.BasicBlock) bool {
	if c.done == nil {
		c.done = map[*ssa.BasicBlock]bool{}
	}

	if c.done[b] {
		return true
	}
	c.done[b] = true

	if len(b.Succs) == 0 {
		return false
	}

	for _, s := range b.Succs {
		if !c.storedInInstrs(s.Instrs) && !c.succs(s) {
			return false
		}
	}

	return true
}

func (c *calledFrom) isRecv(recv, typ types.Type) bool {
	return recv == typ || identical(recv, typ) ||
		c.isRecvInTuple(recv, typ) || c.isRecvInEmbedded(recv, typ)
}

func (c *calledFrom) isRecvInTuple(recv, typ types.Type) bool {
	tuple, _ := typ.(*types.Tuple)
	if tuple == nil {
		return false
	}

	for i := 0; i < tuple.Len(); i++ {
		if c.isRecv(recv, tuple.At(i).Type()) {
			return true
		}
	}

	return false
}

func (c *calledFrom) isRecvInEmbedded(recv, typ types.Type) bool {

	var st *types.Struct
	switch typ := typ.(type) {
	case *types.Struct:
		st = typ
	case *types.Pointer:
		return c.isRecvInEmbedded(recv, typ.Elem())
	case *types.Named:
		return c.isRecvInEmbedded(recv, typ.Underlying())
	default:
		return false
	}

	for i := 0; i < st.NumFields(); i++ {
		field := st.Field(i)
		if !field.Embedded() {
			continue
		}

		ft := field.Type()
		if c.isRecv(recv, ft) {
			return true
		}

		var ptrOrUnptr types.Type
		switch ft := ft.(type) {
		case *types.Pointer:
			// struct { *T } -> T
			ptrOrUnptr = ft.Elem()
		default:
			// struct { T } -> *T
			ptrOrUnptr = types.NewPointer(ft)
		}

		if c.isRecv(recv, ptrOrUnptr) {
			return true
		}
	}

	return false
}

// NotCalledIn checks whether receiver's method is called in a function.
// If there is no methods calling at a path from an instruction
// which type is receiver to all return instruction, NotCalledIn returns these instructions.
func NotCalledIn(f *ssa.Function, receiver types.Type, methods ...*types.Func) []ssa.Instruction {
	return new(CalledChecker).NotIn(f, receiver, methods...)
}

// CalledFrom checks whether receiver's method is called in an instruction
// which belogns to after i-th instructions, or in succsor blocks of b.
// The first result is above value.
// The second result is whether type of i-th instruction does not much receiver
// or matches with ignore cases.
func CalledFrom(b *ssa.BasicBlock, i int, receiver types.Type, methods ...*types.Func) (called, ok bool) {
	return new(CalledChecker).From(b, i, receiver, methods...)
}

// Called returns true when f is called in the instr.
// If recv is not nil, Called also checks the receiver.
func Called(instr ssa.Instruction, recv ssa.Value, f *types.Func) bool {
	return new(CalledChecker).Func(instr, recv, f)
}
