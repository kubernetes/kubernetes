package ir

import (
	"go/types"
)

func (b *builder) buildExits(fn *Function) {
	if obj := fn.Object(); obj != nil {
		switch obj.Pkg().Path() {
		case "runtime":
			switch obj.Name() {
			case "exit":
				fn.NoReturn = AlwaysExits
				return
			case "throw":
				fn.NoReturn = AlwaysExits
				return
			case "Goexit":
				fn.NoReturn = AlwaysUnwinds
				return
			}
		case "go.uber.org/zap":
			switch obj.(*types.Func).FullName() {
			case "(*go.uber.org/zap.Logger).Fatal",
				"(*go.uber.org/zap.SugaredLogger).Fatal",
				"(*go.uber.org/zap.SugaredLogger).Fatalw",
				"(*go.uber.org/zap.SugaredLogger).Fatalf":
				// Technically, this method does not unconditionally exit
				// the process. It dynamically calls a function stored in
				// the logger. If the function is nil, it defaults to
				// os.Exit.
				//
				// The main intent of this method is to terminate the
				// process, and that's what the vast majority of people
				// will use it for. We'll happily accept some false
				// negatives to avoid a lot of false positives.
				fn.NoReturn = AlwaysExits
			case "(*go.uber.org/zap.Logger).Panic",
				"(*go.uber.org/zap.SugaredLogger).Panicw",
				"(*go.uber.org/zap.SugaredLogger).Panicf":
				fn.NoReturn = AlwaysUnwinds
				return
			case "(*go.uber.org/zap.Logger).DPanic",
				"(*go.uber.org/zap.SugaredLogger).DPanicf",
				"(*go.uber.org/zap.SugaredLogger).DPanicw":
				// These methods will only panic in development.
			}
		case "github.com/sirupsen/logrus":
			switch obj.(*types.Func).FullName() {
			case "(*github.com/sirupsen/logrus.Logger).Exit":
				// Technically, this method does not unconditionally exit
				// the process. It dynamically calls a function stored in
				// the logger. If the function is nil, it defaults to
				// os.Exit.
				//
				// The main intent of this method is to terminate the
				// process, and that's what the vast majority of people
				// will use it for. We'll happily accept some false
				// negatives to avoid a lot of false positives.
				fn.NoReturn = AlwaysExits
				return
			case "(*github.com/sirupsen/logrus.Logger).Panic",
				"(*github.com/sirupsen/logrus.Logger).Panicf",
				"(*github.com/sirupsen/logrus.Logger).Panicln":

				// These methods will always panic, but that's not
				// statically known from the code alone, because they
				// take a detour through the generic Log methods.
				fn.NoReturn = AlwaysUnwinds
				return
			case "(*github.com/sirupsen/logrus.Entry).Panicf",
				"(*github.com/sirupsen/logrus.Entry).Panicln":

				// Entry.Panic has an explicit panic, but Panicf and
				// Panicln do not, relying fully on the generic Log
				// method.
				fn.NoReturn = AlwaysUnwinds
				return
			case "(*github.com/sirupsen/logrus.Logger).Log",
				"(*github.com/sirupsen/logrus.Logger).Logf",
				"(*github.com/sirupsen/logrus.Logger).Logln":
				// TODO(dh): we cannot handle these cases. Whether they
				// exit or unwind depends on the level, which is set
				// via the first argument. We don't currently support
				// call-site-specific exit information.
			}
		case "github.com/golang/glog":
			switch obj.(*types.Func).FullName() {
			case "github.com/golang/glog.Exit",
				"github.com/golang/glog.ExitDepth",
				"github.com/golang/glog.Exitf",
				"github.com/golang/glog.Exitln",
				"github.com/golang/glog.Fatal",
				"github.com/golang/glog.FatalDepth",
				"github.com/golang/glog.Fatalf",
				"github.com/golang/glog.Fatalln":
				// all of these call os.Exit after logging
				fn.NoReturn = AlwaysExits
			}
		case "k8s.io/klog":
			switch obj.(*types.Func).FullName() {
			case "k8s.io/klog.Exit",
				"k8s.io/klog.ExitDepth",
				"k8s.io/klog.Exitf",
				"k8s.io/klog.Exitln",
				"k8s.io/klog.Fatal",
				"k8s.io/klog.FatalDepth",
				"k8s.io/klog.Fatalf",
				"k8s.io/klog.Fatalln":
				// all of these call os.Exit after logging
				fn.NoReturn = AlwaysExits
			}
		}
	}

	isRecoverCall := func(instr Instruction) bool {
		if instr, ok := instr.(*Call); ok {
			if builtin, ok := instr.Call.Value.(*Builtin); ok {
				if builtin.Name() == "recover" {
					return true
				}
			}
		}
		return false
	}

	both := NewBlockSet(len(fn.Blocks))
	exits := NewBlockSet(len(fn.Blocks))
	unwinds := NewBlockSet(len(fn.Blocks))
	recovers := false
	for _, u := range fn.Blocks {
		for _, instr := range u.Instrs {
		instrSwitch:
			switch instr := instr.(type) {
			case *Defer:
				if recovers {
					// avoid doing extra work, we already know that this function calls recover
					continue
				}
				call := instr.Call.StaticCallee()
				if call == nil {
					// not a static call, so we can't be sure the
					// deferred call isn't calling recover
					recovers = true
					break
				}
				if call.Package() == fn.Package() {
					b.buildFunction(call)
				}
				if len(call.Blocks) == 0 {
					// external function, we don't know what's
					// happening inside it
					//
					// TODO(dh): this includes functions from
					// imported packages, due to how go/analysis
					// works. We could introduce another fact,
					// like we've done for exiting and unwinding.
					recovers = true
					break
				}
				for _, y := range call.Blocks {
					for _, instr2 := range y.Instrs {
						if isRecoverCall(instr2) {
							recovers = true
							break instrSwitch
						}
					}
				}

			case *Panic:
				both.Add(u)
				unwinds.Add(u)

			case CallInstruction:
				switch instr.(type) {
				case *Defer, *Call:
				default:
					continue
				}
				if instr.Common().IsInvoke() {
					// give up
					return
				}
				var call *Function
				switch instr.Common().Value.(type) {
				case *Function, *MakeClosure:
					call = instr.Common().StaticCallee()
				case *Builtin:
					// the only builtins that affect control flow are
					// panic and recover, and we've already handled
					// those
					continue
				default:
					// dynamic dispatch
					return
				}
				// buildFunction is idempotent. if we're part of a
				// (mutually) recursive call chain, then buildFunction
				// will immediately return, and fn.WillExit will be false.
				if call.Package() == fn.Package() {
					b.buildFunction(call)
				}
				switch call.NoReturn {
				case AlwaysExits:
					both.Add(u)
					exits.Add(u)
				case AlwaysUnwinds:
					both.Add(u)
					unwinds.Add(u)
				case NeverReturns:
					both.Add(u)
				}
			}
		}
	}

	// depth-first search trying to find a path to the exit block that
	// doesn't cross any of the blacklisted blocks
	seen := NewBlockSet(len(fn.Blocks))
	var findPath func(root *BasicBlock, bl *BlockSet) bool
	findPath = func(root *BasicBlock, bl *BlockSet) bool {
		if root == fn.Exit {
			return true
		}
		if seen.Has(root) {
			return false
		}
		if bl.Has(root) {
			return false
		}
		seen.Add(root)
		for _, succ := range root.Succs {
			if findPath(succ, bl) {
				return true
			}
		}
		return false
	}
	findPathEntry := func(root *BasicBlock, bl *BlockSet) bool {
		if bl.Num() == 0 {
			return true
		}
		seen.Clear()
		return findPath(root, bl)
	}

	if !findPathEntry(fn.Blocks[0], exits) {
		fn.NoReturn = AlwaysExits
	} else if !recovers {
		// Only consider unwinding and "never returns" if we don't
		// call recover. If we do call recover, then panics don't
		// bubble up the stack.

		// TODO(dh): the position of the defer matters. If we
		// unconditionally terminate before we defer a recover, then
		// the recover is ineffective.

		if !findPathEntry(fn.Blocks[0], unwinds) {
			fn.NoReturn = AlwaysUnwinds
		} else if !findPathEntry(fn.Blocks[0], both) {
			fn.NoReturn = NeverReturns
		}
	}
}

func (b *builder) addUnreachables(fn *Function) {
	var unreachable *BasicBlock

	for _, bb := range fn.Blocks {
	instrLoop:
		for i, instr := range bb.Instrs {
			if instr, ok := instr.(*Call); ok {
				var call *Function
				switch v := instr.Common().Value.(type) {
				case *Function:
					call = v
				case *MakeClosure:
					call = v.Fn.(*Function)
				}
				if call == nil {
					continue
				}
				if call.Package() == fn.Package() {
					// make sure we have information on all functions in this package
					b.buildFunction(call)
				}
				switch call.NoReturn {
				case AlwaysExits:
					// This call will cause the process to terminate.
					// Remove remaining instructions in the block and
					// replace any control flow with Unreachable.
					for _, succ := range bb.Succs {
						succ.removePred(bb)
					}
					bb.Succs = bb.Succs[:0]

					bb.Instrs = bb.Instrs[:i+1]
					bb.emit(new(Unreachable), instr.Source())
					addEdge(bb, fn.Exit)
					break instrLoop

				case AlwaysUnwinds:
					// This call will cause the goroutine to terminate
					// and defers to run (i.e. a panic or
					// runtime.Goexit). Remove remaining instructions
					// in the block and replace any control flow with
					// an unconditional jump to the exit block.
					for _, succ := range bb.Succs {
						succ.removePred(bb)
					}
					bb.Succs = bb.Succs[:0]

					bb.Instrs = bb.Instrs[:i+1]
					bb.emit(new(Jump), instr.Source())
					addEdge(bb, fn.Exit)
					break instrLoop

				case NeverReturns:
					// This call will either cause the goroutine to
					// terminate, or the process to terminate. Remove
					// remaining instructions in the block and replace
					// any control flow with a conditional jump to
					// either the exit block, or Unreachable.
					for _, succ := range bb.Succs {
						succ.removePred(bb)
					}
					bb.Succs = bb.Succs[:0]

					bb.Instrs = bb.Instrs[:i+1]
					var c Call
					c.Call.Value = &Builtin{
						name: "ir:noreturnWasPanic",
						sig: types.NewSignature(nil,
							types.NewTuple(),
							types.NewTuple(anonVar(types.Typ[types.Bool])),
							false,
						),
					}
					c.setType(types.Typ[types.Bool])

					if unreachable == nil {
						unreachable = fn.newBasicBlock("unreachable")
						unreachable.emit(&Unreachable{}, nil)
						addEdge(unreachable, fn.Exit)
					}

					bb.emit(&c, instr.Source())
					bb.emit(&If{Cond: &c}, instr.Source())
					addEdge(bb, fn.Exit)
					addEdge(bb, unreachable)
					break instrLoop
				}
			}
		}
	}
}
