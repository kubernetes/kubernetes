package starlark

// This file defines the bytecode interpreter.

import (
	"fmt"
	"os"
	"sync/atomic"
	"unsafe"

	"go.starlark.net/internal/compile"
	"go.starlark.net/internal/spell"
	"go.starlark.net/resolve"
	"go.starlark.net/syntax"
)

const vmdebug = false // TODO(adonovan): use a bitfield of specific kinds of error.

// TODO(adonovan):
// - optimize position table.
// - opt: record MaxIterStack during compilation and preallocate the stack.

func (fn *Function) CallInternal(thread *Thread, args Tuple, kwargs []Tuple) (Value, error) {
	// Postcondition: args is not mutated. This is stricter than required by Callable,
	// but allows CALL to avoid a copy.

	if !resolve.AllowRecursion {
		// detect recursion
		for _, fr := range thread.stack[:len(thread.stack)-1] {
			// We look for the same function code,
			// not function value, otherwise the user could
			// defeat the check by writing the Y combinator.
			if frfn, ok := fr.Callable().(*Function); ok && frfn.funcode == fn.funcode {
				return nil, fmt.Errorf("function %s called recursively", fn.Name())
			}
		}
	}

	f := fn.funcode
	fr := thread.frameAt(0)

	// Allocate space for stack and locals.
	// Logically these do not escape from this frame
	// (See https://github.com/golang/go/issues/20533.)
	//
	// This heap allocation looks expensive, but I was unable to get
	// more than 1% real time improvement in a large alloc-heavy
	// benchmark (in which this alloc was 8% of alloc-bytes)
	// by allocating space for 8 Values in each frame, or
	// by allocating stack by slicing an array held by the Thread
	// that is expanded in chunks of min(k, nspace), for k=256 or 1024.
	nlocals := len(f.Locals)
	nspace := nlocals + f.MaxStack
	space := make([]Value, nspace)
	locals := space[:nlocals:nlocals] // local variables, starting with parameters
	stack := space[nlocals:]          // operand stack

	// Digest arguments and set parameters.
	err := setArgs(locals, fn, args, kwargs)
	if err != nil {
		return nil, thread.evalError(err)
	}

	fr.locals = locals

	if vmdebug {
		fmt.Printf("Entering %s @ %s\n", f.Name, f.Position(0))
		fmt.Printf("%d stack, %d locals\n", len(stack), len(locals))
		defer fmt.Println("Leaving ", f.Name)
	}

	// Spill indicated locals to cells.
	// Each cell is a separate alloc to avoid spurious liveness.
	for _, index := range f.Cells {
		locals[index] = &cell{locals[index]}
	}

	// TODO(adonovan): add static check that beneath this point
	// - there is exactly one return statement
	// - there is no redefinition of 'err'.

	var iterstack []Iterator // stack of active iterators

	sp := 0
	var pc uint32
	var result Value
	code := f.Code
loop:
	for {
		thread.steps++
		if thread.steps >= thread.maxSteps {
			if thread.OnMaxSteps != nil {
				thread.OnMaxSteps(thread)
			} else {
				thread.Cancel("too many steps")
			}
		}
		if reason := atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&thread.cancelReason))); reason != nil {
			err = fmt.Errorf("Starlark computation cancelled: %s", *(*string)(reason))
			break loop
		}

		fr.pc = pc

		op := compile.Opcode(code[pc])
		pc++
		var arg uint32
		if op >= compile.OpcodeArgMin {
			// TODO(adonovan): opt: profile this.
			// Perhaps compiling big endian would be less work to decode?
			for s := uint(0); ; s += 7 {
				b := code[pc]
				pc++
				arg |= uint32(b&0x7f) << s
				if b < 0x80 {
					break
				}
			}
		}
		if vmdebug {
			fmt.Fprintln(os.Stderr, stack[:sp]) // very verbose!
			compile.PrintOp(f, fr.pc, op, arg)
		}

		switch op {
		case compile.NOP:
			// nop

		case compile.DUP:
			stack[sp] = stack[sp-1]
			sp++

		case compile.DUP2:
			stack[sp] = stack[sp-2]
			stack[sp+1] = stack[sp-1]
			sp += 2

		case compile.POP:
			sp--

		case compile.EXCH:
			stack[sp-2], stack[sp-1] = stack[sp-1], stack[sp-2]

		case compile.EQL, compile.NEQ, compile.GT, compile.LT, compile.LE, compile.GE:
			op := syntax.Token(op-compile.EQL) + syntax.EQL
			y := stack[sp-1]
			x := stack[sp-2]
			sp -= 2
			ok, err2 := Compare(op, x, y)
			if err2 != nil {
				err = err2
				break loop
			}
			stack[sp] = Bool(ok)
			sp++

		case compile.PLUS,
			compile.MINUS,
			compile.STAR,
			compile.SLASH,
			compile.SLASHSLASH,
			compile.PERCENT,
			compile.AMP,
			compile.PIPE,
			compile.CIRCUMFLEX,
			compile.LTLT,
			compile.GTGT,
			compile.IN:
			binop := syntax.Token(op-compile.PLUS) + syntax.PLUS
			if op == compile.IN {
				binop = syntax.IN // IN token is out of order
			}
			y := stack[sp-1]
			x := stack[sp-2]
			sp -= 2
			z, err2 := Binary(binop, x, y)
			if err2 != nil {
				err = err2
				break loop
			}
			stack[sp] = z
			sp++

		case compile.UPLUS, compile.UMINUS, compile.TILDE:
			var unop syntax.Token
			if op == compile.TILDE {
				unop = syntax.TILDE
			} else {
				unop = syntax.Token(op-compile.UPLUS) + syntax.PLUS
			}
			x := stack[sp-1]
			y, err2 := Unary(unop, x)
			if err2 != nil {
				err = err2
				break loop
			}
			stack[sp-1] = y

		case compile.INPLACE_ADD:
			y := stack[sp-1]
			x := stack[sp-2]
			sp -= 2

			// It's possible that y is not Iterable but
			// nonetheless defines x+y, in which case we
			// should fall back to the general case.
			var z Value
			if xlist, ok := x.(*List); ok {
				if yiter, ok := y.(Iterable); ok {
					if err = xlist.checkMutable("apply += to"); err != nil {
						break loop
					}
					listExtend(xlist, yiter)
					z = xlist
				}
			}
			if z == nil {
				z, err = Binary(syntax.PLUS, x, y)
				if err != nil {
					break loop
				}
			}

			stack[sp] = z
			sp++

		case compile.NONE:
			stack[sp] = None
			sp++

		case compile.TRUE:
			stack[sp] = True
			sp++

		case compile.FALSE:
			stack[sp] = False
			sp++

		case compile.MANDATORY:
			stack[sp] = mandatory{}
			sp++

		case compile.JMP:
			pc = arg

		case compile.CALL, compile.CALL_VAR, compile.CALL_KW, compile.CALL_VAR_KW:
			var kwargs Value
			if op == compile.CALL_KW || op == compile.CALL_VAR_KW {
				kwargs = stack[sp-1]
				sp--
			}

			var args Value
			if op == compile.CALL_VAR || op == compile.CALL_VAR_KW {
				args = stack[sp-1]
				sp--
			}

			// named args (pairs)
			var kvpairs []Tuple
			if nkvpairs := int(arg & 0xff); nkvpairs > 0 {
				kvpairs = make([]Tuple, 0, nkvpairs)
				kvpairsAlloc := make(Tuple, 2*nkvpairs) // allocate a single backing array
				sp -= 2 * nkvpairs
				for i := 0; i < nkvpairs; i++ {
					pair := kvpairsAlloc[:2:2]
					kvpairsAlloc = kvpairsAlloc[2:]
					pair[0] = stack[sp+2*i]   // name
					pair[1] = stack[sp+2*i+1] // value
					kvpairs = append(kvpairs, pair)
				}
			}
			if kwargs != nil {
				// Add key/value items from **kwargs dictionary.
				dict, ok := kwargs.(IterableMapping)
				if !ok {
					err = fmt.Errorf("argument after ** must be a mapping, not %s", kwargs.Type())
					break loop
				}
				items := dict.Items()
				for _, item := range items {
					if _, ok := item[0].(String); !ok {
						err = fmt.Errorf("keywords must be strings, not %s", item[0].Type())
						break loop
					}
				}
				if len(kvpairs) == 0 {
					kvpairs = items
				} else {
					kvpairs = append(kvpairs, items...)
				}
			}

			// positional args
			var positional Tuple
			if npos := int(arg >> 8); npos > 0 {
				positional = stack[sp-npos : sp]
				sp -= npos

				// Copy positional arguments into a new array,
				// unless the callee is another Starlark function,
				// in which case it can be trusted not to mutate them.
				if _, ok := stack[sp-1].(*Function); !ok || args != nil {
					positional = append(Tuple(nil), positional...)
				}
			}
			if args != nil {
				// Add elements from *args sequence.
				iter := Iterate(args)
				if iter == nil {
					err = fmt.Errorf("argument after * must be iterable, not %s", args.Type())
					break loop
				}
				var elem Value
				for iter.Next(&elem) {
					positional = append(positional, elem)
				}
				iter.Done()
			}

			function := stack[sp-1]

			if vmdebug {
				fmt.Printf("VM call %s args=%s kwargs=%s @%s\n",
					function, positional, kvpairs, f.Position(fr.pc))
			}

			thread.endProfSpan()
			z, err2 := Call(thread, function, positional, kvpairs)
			thread.beginProfSpan()
			if err2 != nil {
				err = err2
				break loop
			}
			if vmdebug {
				fmt.Printf("Resuming %s @ %s\n", f.Name, f.Position(0))
			}
			stack[sp-1] = z

		case compile.ITERPUSH:
			x := stack[sp-1]
			sp--
			iter := Iterate(x)
			if iter == nil {
				err = fmt.Errorf("%s value is not iterable", x.Type())
				break loop
			}
			iterstack = append(iterstack, iter)

		case compile.ITERJMP:
			iter := iterstack[len(iterstack)-1]
			if iter.Next(&stack[sp]) {
				sp++
			} else {
				pc = arg
			}

		case compile.ITERPOP:
			n := len(iterstack) - 1
			iterstack[n].Done()
			iterstack = iterstack[:n]

		case compile.NOT:
			stack[sp-1] = !stack[sp-1].Truth()

		case compile.RETURN:
			result = stack[sp-1]
			break loop

		case compile.SETINDEX:
			z := stack[sp-1]
			y := stack[sp-2]
			x := stack[sp-3]
			sp -= 3
			err = setIndex(x, y, z)
			if err != nil {
				break loop
			}

		case compile.INDEX:
			y := stack[sp-1]
			x := stack[sp-2]
			sp -= 2
			z, err2 := getIndex(x, y)
			if err2 != nil {
				err = err2
				break loop
			}
			stack[sp] = z
			sp++

		case compile.ATTR:
			x := stack[sp-1]
			name := f.Prog.Names[arg]
			y, err2 := getAttr(x, name)
			if err2 != nil {
				err = err2
				break loop
			}
			stack[sp-1] = y

		case compile.SETFIELD:
			y := stack[sp-1]
			x := stack[sp-2]
			sp -= 2
			name := f.Prog.Names[arg]
			if err2 := setField(x, name, y); err2 != nil {
				err = err2
				break loop
			}

		case compile.MAKEDICT:
			stack[sp] = new(Dict)
			sp++

		case compile.SETDICT, compile.SETDICTUNIQ:
			dict := stack[sp-3].(*Dict)
			k := stack[sp-2]
			v := stack[sp-1]
			sp -= 3
			oldlen := dict.Len()
			if err2 := dict.SetKey(k, v); err2 != nil {
				err = err2
				break loop
			}
			if op == compile.SETDICTUNIQ && dict.Len() == oldlen {
				err = fmt.Errorf("duplicate key: %v", k)
				break loop
			}

		case compile.APPEND:
			elem := stack[sp-1]
			list := stack[sp-2].(*List)
			sp -= 2
			list.elems = append(list.elems, elem)

		case compile.SLICE:
			x := stack[sp-4]
			lo := stack[sp-3]
			hi := stack[sp-2]
			step := stack[sp-1]
			sp -= 4
			res, err2 := slice(x, lo, hi, step)
			if err2 != nil {
				err = err2
				break loop
			}
			stack[sp] = res
			sp++

		case compile.UNPACK:
			n := int(arg)
			iterable := stack[sp-1]
			sp--
			iter := Iterate(iterable)
			if iter == nil {
				err = fmt.Errorf("got %s in sequence assignment", iterable.Type())
				break loop
			}
			i := 0
			sp += n
			for i < n && iter.Next(&stack[sp-1-i]) {
				i++
			}
			var dummy Value
			if iter.Next(&dummy) {
				// NB: Len may return -1 here in obscure cases.
				err = fmt.Errorf("too many values to unpack (got %d, want %d)", Len(iterable), n)
				break loop
			}
			iter.Done()
			if i < n {
				err = fmt.Errorf("too few values to unpack (got %d, want %d)", i, n)
				break loop
			}

		case compile.CJMP:
			if stack[sp-1].Truth() {
				pc = arg
			}
			sp--

		case compile.CONSTANT:
			stack[sp] = fn.module.constants[arg]
			sp++

		case compile.MAKETUPLE:
			n := int(arg)
			tuple := make(Tuple, n)
			sp -= n
			copy(tuple, stack[sp:])
			stack[sp] = tuple
			sp++

		case compile.MAKELIST:
			n := int(arg)
			elems := make([]Value, n)
			sp -= n
			copy(elems, stack[sp:])
			stack[sp] = NewList(elems)
			sp++

		case compile.MAKEFUNC:
			funcode := f.Prog.Functions[arg]
			tuple := stack[sp-1].(Tuple)
			n := len(tuple) - len(funcode.Freevars)
			defaults := tuple[:n:n]
			freevars := tuple[n:]
			stack[sp-1] = &Function{
				funcode:  funcode,
				module:   fn.module,
				defaults: defaults,
				freevars: freevars,
			}

		case compile.LOAD:
			n := int(arg)
			module := string(stack[sp-1].(String))
			sp--

			if thread.Load == nil {
				err = fmt.Errorf("load not implemented by this application")
				break loop
			}

			thread.endProfSpan()
			dict, err2 := thread.Load(thread, module)
			thread.beginProfSpan()
			if err2 != nil {
				err = wrappedError{
					msg:   fmt.Sprintf("cannot load %s: %v", module, err2),
					cause: err2,
				}
				break loop
			}

			for i := 0; i < n; i++ {
				from := string(stack[sp-1-i].(String))
				v, ok := dict[from]
				if !ok {
					err = fmt.Errorf("load: name %s not found in module %s", from, module)
					if n := spell.Nearest(from, dict.Keys()); n != "" {
						err = fmt.Errorf("%s (did you mean %s?)", err, n)
					}
					break loop
				}
				stack[sp-1-i] = v
			}

		case compile.SETLOCAL:
			locals[arg] = stack[sp-1]
			sp--

		case compile.SETLOCALCELL:
			locals[arg].(*cell).v = stack[sp-1]
			sp--

		case compile.SETGLOBAL:
			fn.module.globals[arg] = stack[sp-1]
			sp--

		case compile.LOCAL:
			x := locals[arg]
			if x == nil {
				err = fmt.Errorf("local variable %s referenced before assignment", f.Locals[arg].Name)
				break loop
			}
			stack[sp] = x
			sp++

		case compile.FREE:
			stack[sp] = fn.freevars[arg]
			sp++

		case compile.LOCALCELL:
			v := locals[arg].(*cell).v
			if v == nil {
				err = fmt.Errorf("local variable %s referenced before assignment", f.Locals[arg].Name)
				break loop
			}
			stack[sp] = v
			sp++

		case compile.FREECELL:
			v := fn.freevars[arg].(*cell).v
			if v == nil {
				err = fmt.Errorf("local variable %s referenced before assignment", f.Freevars[arg].Name)
				break loop
			}
			stack[sp] = v
			sp++

		case compile.GLOBAL:
			x := fn.module.globals[arg]
			if x == nil {
				err = fmt.Errorf("global variable %s referenced before assignment", f.Prog.Globals[arg].Name)
				break loop
			}
			stack[sp] = x
			sp++

		case compile.PREDECLARED:
			name := f.Prog.Names[arg]
			x := fn.module.predeclared[name]
			if x == nil {
				err = fmt.Errorf("internal error: predeclared variable %s is uninitialized", name)
				break loop
			}
			stack[sp] = x
			sp++

		case compile.UNIVERSAL:
			stack[sp] = Universe[f.Prog.Names[arg]]
			sp++

		default:
			err = fmt.Errorf("unimplemented: %s", op)
			break loop
		}
	}

	// ITERPOP the rest of the iterator stack.
	for _, iter := range iterstack {
		iter.Done()
	}

	fr.locals = nil

	return result, err
}

type wrappedError struct {
	msg   string
	cause error
}

func (e wrappedError) Error() string {
	return e.msg
}

// Implements the xerrors.Wrapper interface
// https://godoc.org/golang.org/x/xerrors#Wrapper
func (e wrappedError) Unwrap() error {
	return e.cause
}

// mandatory is a sentinel value used in a function's defaults tuple
// to indicate that a (keyword-only) parameter is mandatory.
type mandatory struct{}

func (mandatory) String() string        { return "mandatory" }
func (mandatory) Type() string          { return "mandatory" }
func (mandatory) Freeze()               {} // immutable
func (mandatory) Truth() Bool           { return False }
func (mandatory) Hash() (uint32, error) { return 0, nil }

// A cell is a box containing a Value.
// Local variables marked as cells hold their value indirectly
// so that they may be shared by outer and inner nested functions.
// Cells are always accessed using indirect {FREE,LOCAL,SETLOCAL}CELL instructions.
// The FreeVars tuple contains only cells.
// The FREE instruction always yields a cell.
type cell struct{ v Value }

func (c *cell) String() string { return "cell" }
func (c *cell) Type() string   { return "cell" }
func (c *cell) Freeze() {
	if c.v != nil {
		c.v.Freeze()
	}
}
func (c *cell) Truth() Bool           { panic("unreachable") }
func (c *cell) Hash() (uint32, error) { panic("unreachable") }
