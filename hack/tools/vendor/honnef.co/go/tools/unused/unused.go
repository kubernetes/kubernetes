package unused

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io"
	"reflect"
	"strings"

	"honnef.co/go/tools/analysis/code"
	"honnef.co/go/tools/analysis/facts"
	"honnef.co/go/tools/analysis/lint"
	"honnef.co/go/tools/analysis/report"
	"honnef.co/go/tools/go/ast/astutil"
	"honnef.co/go/tools/go/ir"
	"honnef.co/go/tools/go/types/typeutil"
	"honnef.co/go/tools/internal/passes/buildir"
	"honnef.co/go/tools/unused/typemap"

	"golang.org/x/tools/go/analysis"
)

var Debug io.Writer

// The graph we construct omits nodes along a path that do not
// contribute any new information to the solution. For example, the
// full graph for a function with a receiver would be Func ->
// Signature -> Var -> Type. However, since signatures cannot be
// unused, and receivers are always considered used, we can compact
// the graph down to Func -> Type. This makes the graph smaller, but
// harder to debug.

// TODO(dh): conversions between structs mark fields as used, but the
// conversion itself isn't part of that subgraph. even if the function
// containing the conversion is unused, the fields will be marked as
// used.

// TODO(dh): we cannot observe function calls in assembly files.

/*

- packages use:
  - (1.1) exported named types
  - (1.2) exported functions
  - (1.3) exported variables
  - (1.4) exported constants
  - (1.5) init functions
  - (1.6) functions exported to cgo
  - (1.7) the main function iff in the main package
  - (1.8) symbols linked via go:linkname

- named types use:
  - (2.1) exported methods
  - (2.2) the type they're based on
  - (2.3) all their aliases. we can't easily track uses of aliases
    because go/types turns them into uses of the aliased types. assume
    that if a type is used, so are all of its aliases.
  - (2.4) the pointer type. this aids with eagerly implementing
    interfaces. if a method that implements an interface is defined on
    a pointer receiver, and the pointer type is never used, but the
    named type is, then we still want to mark the method as used.

- variables and constants use:
  - their types

- functions use:
  - (4.1) all their arguments, return parameters and receivers
  - (4.2) anonymous functions defined beneath them
  - (4.3) closures and bound methods.
    this implements a simplified model where a function is used merely by being referenced, even if it is never called.
    that way we don't have to keep track of closures escaping functions.
  - (4.4) functions they return. we assume that someone else will call the returned function
  - (4.5) functions/interface methods they call
  - types they instantiate or convert to
  - (4.7) fields they access
  - (4.8) types of all instructions
  - (4.9) package-level variables they assign to iff in tests (sinks for benchmarks)

- conversions use:
  - (5.1) when converting between two equivalent structs, the fields in
    either struct use each other. the fields are relevant for the
    conversion, but only if the fields are also accessed outside the
    conversion.
  - (5.2) when converting to or from unsafe.Pointer, mark all fields as used.

- structs use:
  - (6.1) fields of type NoCopy sentinel
  - (6.2) exported fields
  - (6.3) embedded fields that help implement interfaces (either fully implements it, or contributes required methods) (recursively)
  - (6.4) embedded fields that have exported methods (recursively)
  - (6.5) embedded structs that have exported fields (recursively)

- (7.1) field accesses use fields
- (7.2) fields use their types

- (8.0) How we handle interfaces:
  - (8.1) We do not technically care about interfaces that only consist of
    exported methods. Exported methods on concrete types are always
    marked as used.
  - Any concrete type implements all known interfaces. Even if it isn't
    assigned to any interfaces in our code, the user may receive a value
    of the type and expect to pass it back to us through an interface.

    Concrete types use their methods that implement interfaces. If the
    type is used, it uses those methods. Otherwise, it doesn't. This
    way, types aren't incorrectly marked reachable through the edge
    from method to type.

  - (8.3) All interface methods are marked as used, even if they never get
    called. This is to accommodate sum types (unexported interface
    method that must exist but never gets called.)

  - (8.4) All embedded interfaces are marked as used. This is an
    extension of 8.3, but we have to explicitly track embedded
    interfaces because in a chain C->B->A, B wouldn't be marked as
    used by 8.3 just because it contributes A's methods to C.

- Inherent uses:
  - thunks and other generated wrappers call the real function
  - (9.2) variables use their types
  - (9.3) types use their underlying and element types
  - (9.4) conversions use the type they convert to
  - (9.5) instructions use their operands
  - (9.6) instructions use their operands' types
  - (9.7) variable _reads_ use variables, writes do not, except in tests
  - (9.8) runtime functions that may be called from user code via the compiler


- const groups:
  (10.1) if one constant out of a block of constants is used, mark all
  of them used. a lot of the time, unused constants exist for the sake
  of completeness. See also
  https://github.com/dominikh/go-tools/issues/365


- (11.1) anonymous struct types use all their fields. we cannot
  deduplicate struct types, as that leads to order-dependent
  reports. we can't not deduplicate struct types while still
  tracking fields, because then each instance of the unnamed type in
  the data flow chain will get its own fields, causing false
  positives. Thus, we only accurately track fields of named struct
  types, and assume that unnamed struct types use all their fields.

*/

func assert(b bool) {
	if !b {
		panic("failed assertion")
	}
}

// /usr/lib/go/src/runtime/proc.go:433:6: func badmorestackg0 is unused (U1000)

// Functions defined in the Go runtime that may be called through
// compiler magic or via assembly.
var runtimeFuncs = map[string]bool{
	// The first part of the list is copied from
	// cmd/compile/internal/gc/builtin.go, var runtimeDecls
	"newobject":            true,
	"panicindex":           true,
	"panicslice":           true,
	"panicdivide":          true,
	"panicmakeslicelen":    true,
	"throwinit":            true,
	"panicwrap":            true,
	"gopanic":              true,
	"gorecover":            true,
	"goschedguarded":       true,
	"printbool":            true,
	"printfloat":           true,
	"printint":             true,
	"printhex":             true,
	"printuint":            true,
	"printcomplex":         true,
	"printstring":          true,
	"printpointer":         true,
	"printiface":           true,
	"printeface":           true,
	"printslice":           true,
	"printnl":              true,
	"printsp":              true,
	"printlock":            true,
	"printunlock":          true,
	"concatstring2":        true,
	"concatstring3":        true,
	"concatstring4":        true,
	"concatstring5":        true,
	"concatstrings":        true,
	"cmpstring":            true,
	"intstring":            true,
	"slicebytetostring":    true,
	"slicebytetostringtmp": true,
	"slicerunetostring":    true,
	"stringtoslicebyte":    true,
	"stringtoslicerune":    true,
	"slicecopy":            true,
	"slicestringcopy":      true,
	"decoderune":           true,
	"countrunes":           true,
	"convI2I":              true,
	"convT16":              true,
	"convT32":              true,
	"convT64":              true,
	"convTstring":          true,
	"convTslice":           true,
	"convT2E":              true,
	"convT2Enoptr":         true,
	"convT2I":              true,
	"convT2Inoptr":         true,
	"assertE2I":            true,
	"assertE2I2":           true,
	"assertI2I":            true,
	"assertI2I2":           true,
	"panicdottypeE":        true,
	"panicdottypeI":        true,
	"panicnildottype":      true,
	"ifaceeq":              true,
	"efaceeq":              true,
	"fastrand":             true,
	"makemap64":            true,
	"makemap":              true,
	"makemap_small":        true,
	"mapaccess1":           true,
	"mapaccess1_fast32":    true,
	"mapaccess1_fast64":    true,
	"mapaccess1_faststr":   true,
	"mapaccess1_fat":       true,
	"mapaccess2":           true,
	"mapaccess2_fast32":    true,
	"mapaccess2_fast64":    true,
	"mapaccess2_faststr":   true,
	"mapaccess2_fat":       true,
	"mapassign":            true,
	"mapassign_fast32":     true,
	"mapassign_fast32ptr":  true,
	"mapassign_fast64":     true,
	"mapassign_fast64ptr":  true,
	"mapassign_faststr":    true,
	"mapiterinit":          true,
	"mapdelete":            true,
	"mapdelete_fast32":     true,
	"mapdelete_fast64":     true,
	"mapdelete_faststr":    true,
	"mapiternext":          true,
	"mapclear":             true,
	"makechan64":           true,
	"makechan":             true,
	"chanrecv1":            true,
	"chanrecv2":            true,
	"chansend1":            true,
	"closechan":            true,
	"writeBarrier":         true,
	"typedmemmove":         true,
	"typedmemclr":          true,
	"typedslicecopy":       true,
	"selectnbsend":         true,
	"selectnbrecv":         true,
	"selectnbrecv2":        true,
	"selectsetpc":          true,
	"selectgo":             true,
	"block":                true,
	"makeslice":            true,
	"makeslice64":          true,
	"growslice":            true,
	"memmove":              true,
	"memclrNoHeapPointers": true,
	"memclrHasPointers":    true,
	"memequal":             true,
	"memequal8":            true,
	"memequal16":           true,
	"memequal32":           true,
	"memequal64":           true,
	"memequal128":          true,
	"int64div":             true,
	"uint64div":            true,
	"int64mod":             true,
	"uint64mod":            true,
	"float64toint64":       true,
	"float64touint64":      true,
	"float64touint32":      true,
	"int64tofloat64":       true,
	"uint64tofloat64":      true,
	"uint32tofloat64":      true,
	"complex128div":        true,
	"racefuncenter":        true,
	"racefuncenterfp":      true,
	"racefuncexit":         true,
	"raceread":             true,
	"racewrite":            true,
	"racereadrange":        true,
	"racewriterange":       true,
	"msanread":             true,
	"msanwrite":            true,
	"x86HasPOPCNT":         true,
	"x86HasSSE41":          true,
	"arm64HasATOMICS":      true,

	// The second part of the list is extracted from assembly code in
	// the standard library, with the exception of the runtime package itself
	"abort":                 true,
	"aeshashbody":           true,
	"args":                  true,
	"asminit":               true,
	"badctxt":               true,
	"badmcall2":             true,
	"badmcall":              true,
	"badmorestackg0":        true,
	"badmorestackgsignal":   true,
	"badsignal2":            true,
	"callbackasm1":          true,
	"callCfunction":         true,
	"cgocallback_gofunc":    true,
	"cgocallbackg":          true,
	"checkgoarm":            true,
	"check":                 true,
	"debugCallCheck":        true,
	"debugCallWrap":         true,
	"emptyfunc":             true,
	"entersyscall":          true,
	"exit":                  true,
	"exits":                 true,
	"exitsyscall":           true,
	"externalthreadhandler": true,
	"findnull":              true,
	"goexit1":               true,
	"gostring":              true,
	"i386_set_ldt":          true,
	"_initcgo":              true,
	"init_thread_tls":       true,
	"ldt0setup":             true,
	"libpreinit":            true,
	"load_g":                true,
	"morestack":             true,
	"mstart":                true,
	"nacl_sysinfo":          true,
	"nanotimeQPC":           true,
	"nanotime":              true,
	"newosproc0":            true,
	"newproc":               true,
	"newstack":              true,
	"noted":                 true,
	"nowQPC":                true,
	"osinit":                true,
	"printf":                true,
	"racecallback":          true,
	"reflectcallmove":       true,
	"reginit":               true,
	"rt0_go":                true,
	"save_g":                true,
	"schedinit":             true,
	"setldt":                true,
	"settls":                true,
	"sighandler":            true,
	"sigprofNonGo":          true,
	"sigtrampgo":            true,
	"_sigtramp":             true,
	"sigtramp":              true,
	"stackcheck":            true,
	"syscall_chdir":         true,
	"syscall_chroot":        true,
	"syscall_close":         true,
	"syscall_dup2":          true,
	"syscall_execve":        true,
	"syscall_exit":          true,
	"syscall_fcntl":         true,
	"syscall_forkx":         true,
	"syscall_gethostname":   true,
	"syscall_getpid":        true,
	"syscall_ioctl":         true,
	"syscall_pipe":          true,
	"syscall_rawsyscall6":   true,
	"syscall_rawSyscall6":   true,
	"syscall_rawsyscall":    true,
	"syscall_RawSyscall":    true,
	"syscall_rawsysvicall6": true,
	"syscall_setgid":        true,
	"syscall_setgroups":     true,
	"syscall_setpgid":       true,
	"syscall_setsid":        true,
	"syscall_setuid":        true,
	"syscall_syscall6":      true,
	"syscall_syscall":       true,
	"syscall_Syscall":       true,
	"syscall_sysvicall6":    true,
	"syscall_wait4":         true,
	"syscall_write":         true,
	"traceback":             true,
	"tstart":                true,
	"usplitR0":              true,
	"wbBufFlush":            true,
	"write":                 true,
}

type pkg struct {
	Fset       *token.FileSet
	Files      []*ast.File
	Pkg        *types.Package
	TypesInfo  *types.Info
	TypesSizes types.Sizes
	IR         *ir.Package
	SrcFuncs   []*ir.Function
	Directives []lint.Directive
}

// TODO(dh): should we return a map instead of two slices?
type Result struct {
	Used   []types.Object
	Unused []types.Object
}

type SerializedResult struct {
	Used   []SerializedObject
	Unused []SerializedObject
}

var Analyzer = &lint.Analyzer{
	Doc: &lint.Documentation{
		Title: "Unused code",
	},
	Analyzer: &analysis.Analyzer{
		Name:       "U1000",
		Doc:        "Unused code",
		Run:        run,
		Requires:   []*analysis.Analyzer{buildir.Analyzer, facts.Generated, facts.Directives},
		ResultType: reflect.TypeOf(Result{}),
	},
}

type SerializedObject struct {
	Name            string
	Position        token.Position
	DisplayPosition token.Position
	Kind            string
	InGenerated     bool
}

func typString(obj types.Object) string {
	switch obj := obj.(type) {
	case *types.Func:
		return "func"
	case *types.Var:
		if obj.IsField() {
			return "field"
		}
		return "var"
	case *types.Const:
		return "const"
	case *types.TypeName:
		return "type"
	default:
		return "identifier"
	}
}

func Serialize(pass *analysis.Pass, res Result, fset *token.FileSet) SerializedResult {
	// OPT(dh): there's no point in serializing Used objects that are
	// always used, such as exported names, blank identifiers, or
	// anonymous struct fields. Used only exists to overrule Unused of
	// a different package. If something can never be unused, then its
	// presence in Used is useless.
	//
	// I'm not sure if this should happen when serializing, or when
	// returning Result.

	out := SerializedResult{
		Used:   make([]SerializedObject, len(res.Used)),
		Unused: make([]SerializedObject, len(res.Unused)),
	}
	for i, obj := range res.Used {
		out.Used[i] = serializeObject(pass, fset, obj)
	}
	for i, obj := range res.Unused {
		out.Unused[i] = serializeObject(pass, fset, obj)
	}
	return out
}

func serializeObject(pass *analysis.Pass, fset *token.FileSet, obj types.Object) SerializedObject {
	name := obj.Name()
	if sig, ok := obj.Type().(*types.Signature); ok && sig.Recv() != nil {
		switch sig.Recv().Type().(type) {
		case *types.Named, *types.Pointer:
			typ := types.TypeString(sig.Recv().Type(), func(*types.Package) string { return "" })
			if len(typ) > 0 && typ[0] == '*' {
				name = fmt.Sprintf("(%s).%s", typ, obj.Name())
			} else if len(typ) > 0 {
				name = fmt.Sprintf("%s.%s", typ, obj.Name())
			}
		}
	}
	return SerializedObject{
		Name:            name,
		Position:        fset.PositionFor(obj.Pos(), false),
		DisplayPosition: report.DisplayPosition(fset, obj.Pos()),
		Kind:            typString(obj),
		InGenerated:     code.IsGenerated(pass, obj.Pos()),
	}
}

func debugf(f string, v ...interface{}) {
	if Debug != nil {
		fmt.Fprintf(Debug, f, v...)
	}
}

func run(pass *analysis.Pass) (interface{}, error) {
	irpkg := pass.ResultOf[buildir.Analyzer].(*buildir.IR)
	dirs := pass.ResultOf[facts.Directives].([]lint.Directive)
	pkg := &pkg{
		Fset:       pass.Fset,
		Files:      pass.Files,
		Pkg:        pass.Pkg,
		TypesInfo:  pass.TypesInfo,
		TypesSizes: pass.TypesSizes,
		IR:         irpkg.Pkg,
		SrcFuncs:   irpkg.SrcFuncs,
		Directives: dirs,
	}

	g := newGraph()
	g.entry(pkg)
	used, unused := results(g)

	if Debug != nil {
		debugNode := func(n *node) {
			if n.obj == nil {
				debugf("n%d [label=\"Root\"];\n", n.id)
			} else {
				color := "red"
				if n.seen {
					color = "green"
				}
				debugf("n%d [label=%q, color=%q];\n", n.id, fmt.Sprintf("(%T) %s", n.obj, n.obj), color)
			}
			for _, e := range n.used {
				for i := edgeKind(1); i < 64; i++ {
					if e.kind.is(1 << i) {
						debugf("n%d -> n%d [label=%q];\n", n.id, e.node.id, edgeKind(1<<i))
					}
				}
			}
		}

		debugf("digraph{\n")
		debugNode(g.Root)
		for _, v := range g.Nodes {
			debugNode(v)
		}
		g.TypeNodes.Iterate(func(key types.Type, value interface{}) {
			debugNode(value.(*node))
		})

		debugf("}\n")
	}

	return Result{Used: used, Unused: unused}, nil
}

func results(g *graph) (used, unused []types.Object) {
	g.color(g.Root)
	g.TypeNodes.Iterate(func(_ types.Type, value interface{}) {
		node := value.(*node)
		if node.seen {
			return
		}
		switch obj := node.obj.(type) {
		case *types.Struct:
			for i := 0; i < obj.NumFields(); i++ {
				if node, ok := g.nodeMaybe(obj.Field(i)); ok {
					node.quiet = true
				}
			}
		case *types.Interface:
			for i := 0; i < obj.NumExplicitMethods(); i++ {
				m := obj.ExplicitMethod(i)
				if node, ok := g.nodeMaybe(m); ok {
					node.quiet = true
				}
			}
		}
	})

	// OPT(dh): can we find meaningful initial capacities for the used and unused slices?

	for _, n := range g.Nodes {
		if obj, ok := n.obj.(types.Object); ok {
			switch obj := obj.(type) {
			case *types.Var:
				// don't report unnamed variables (interface embedding)
				if obj.Name() == "" && obj.IsField() {
					continue
				}
			case types.Object:
				if obj.Name() == "_" {
					continue
				}
			}

			if obj.Pkg() != nil {
				if n.seen {
					used = append(used, obj)
				} else if !n.quiet {
					if obj.Pkg() != g.pkg.Pkg {
						continue
					}
					unused = append(unused, obj)
				}
			}
		}
	}

	return used, unused
}

type graph struct {
	Root      *node
	seenTypes typemap.Map

	TypeNodes typemap.Map
	Nodes     map[interface{}]*node

	// context
	pkg         *pkg
	seenFns     map[*ir.Function]struct{}
	nodeCounter uint64
}

func newGraph() *graph {
	g := &graph{
		Nodes:   map[interface{}]*node{},
		seenFns: map[*ir.Function]struct{}{},
	}
	g.Root = g.newNode(nil)
	return g
}

func (g *graph) color(root *node) {
	if root.seen {
		return
	}
	root.seen = true
	for _, e := range root.used {
		g.color(e.node)
	}
}

type constGroup struct {
	// give the struct a size to get unique pointers
	_ byte
}

func (constGroup) String() string { return "const group" }

type edge struct {
	node *node
	kind edgeKind
}

type node struct {
	obj interface{}
	id  uint64

	// OPT(dh): evaluate using a map instead of a slice to avoid
	// duplicate edges.
	used []edge

	// set during final graph walk if node is reachable
	seen  bool
	quiet bool
}

func (g *graph) nodeMaybe(obj types.Object) (*node, bool) {
	if node, ok := g.Nodes[obj]; ok {
		return node, true
	}
	return nil, false
}

func (g *graph) node(obj interface{}) (n *node, new bool) {
	switch obj := obj.(type) {
	case types.Type:
		if v := g.TypeNodes.At(obj); v != nil {
			return v.(*node), false
		}
		n = g.newNode(obj)
		g.TypeNodes.Set(obj, n)
		return n, true
	case types.Object:
		// OPT(dh): the types.Object and default cases are identical
		if node, ok := g.Nodes[obj]; ok {
			return node, false
		}

		n = g.newNode(obj)
		g.Nodes[obj] = n
		return n, true
	default:
		if node, ok := g.Nodes[obj]; ok {
			return node, false
		}

		n = g.newNode(obj)
		g.Nodes[obj] = n
		return n, true
	}
}

func (g *graph) newNode(obj interface{}) *node {
	g.nodeCounter++
	return &node{
		obj: obj,
		id:  g.nodeCounter,
	}
}

func (n *node) use(n2 *node, kind edgeKind) {
	assert(n2 != nil)
	n.used = append(n.used, edge{node: n2, kind: kind})
}

// isIrrelevant reports whether an object's presence in the graph is
// of any relevance. A lot of objects will never have outgoing edges,
// nor meaningful incoming ones. Examples are basic types and empty
// signatures, among many others.
//
// Dropping these objects should have no effect on correctness, but
// may improve performance. It also helps with debugging, as it
// greatly reduces the size of the graph.
func isIrrelevant(obj interface{}) bool {
	if obj, ok := obj.(types.Object); ok {
		switch obj := obj.(type) {
		case *types.Var:
			if obj.IsField() {
				// We need to track package fields
				return false
			}
			if obj.Pkg() != nil && obj.Parent() == obj.Pkg().Scope() {
				// We need to track package-level variables
				return false
			}
			return isIrrelevant(obj.Type())
		default:
			return false
		}
	}
	if T, ok := obj.(types.Type); ok {
		switch T := T.(type) {
		case *types.Array:
			return isIrrelevant(T.Elem())
		case *types.Slice:
			return isIrrelevant(T.Elem())
		case *types.Basic:
			return true
		case *types.Tuple:
			for i := 0; i < T.Len(); i++ {
				if !isIrrelevant(T.At(i).Type()) {
					return false
				}
			}
			return true
		case *types.Signature:
			if T.Recv() != nil {
				return false
			}
			for i := 0; i < T.Params().Len(); i++ {
				if !isIrrelevant(T.Params().At(i)) {
					return false
				}
			}
			for i := 0; i < T.Results().Len(); i++ {
				if !isIrrelevant(T.Results().At(i)) {
					return false
				}
			}
			return true
		case *types.Interface:
			return T.NumMethods() == 0 && T.NumEmbeddeds() == 0
		case *types.Pointer:
			return isIrrelevant(T.Elem())
		case *types.Map:
			return isIrrelevant(T.Key()) && isIrrelevant(T.Elem())
		case *types.Struct:
			return T.NumFields() == 0
		case *types.Chan:
			return isIrrelevant(T.Elem())
		default:
			return false
		}
	}
	return false
}

func (g *graph) see(obj interface{}) *node {
	if isIrrelevant(obj) {
		return nil
	}

	assert(obj != nil)
	// add new node to graph
	node, _ := g.node(obj)
	return node
}

func (g *graph) use(used, by interface{}, kind edgeKind) {
	if isIrrelevant(used) {
		return
	}

	assert(used != nil)
	if obj, ok := by.(types.Object); ok && obj.Pkg() != nil {
		if obj.Pkg() != g.pkg.Pkg {
			return
		}
	}
	usedNode, new := g.node(used)
	assert(!new)
	if by == nil {
		g.Root.use(usedNode, kind)
	} else {
		byNode, new := g.node(by)
		assert(!new)
		byNode.use(usedNode, kind)
	}
}

func (g *graph) seeAndUse(used, by interface{}, kind edgeKind) *node {
	n := g.see(used)
	g.use(used, by, kind)
	return n
}

func (g *graph) entry(pkg *pkg) {
	g.pkg = pkg
	scopes := map[*types.Scope]*ir.Function{}
	for _, fn := range pkg.SrcFuncs {
		if fn.Object() != nil {
			scope := fn.Object().(*types.Func).Scope()
			scopes[scope] = fn
		}
	}

	for _, f := range pkg.Files {
		for _, cg := range f.Comments {
			for _, c := range cg.List {
				if strings.HasPrefix(c.Text, "//go:linkname ") {
					// FIXME(dh): we're looking at all comments. The
					// compiler only looks at comments in the
					// left-most column. The intention probably is to
					// only look at top-level comments.

					// (1.8) packages use symbols linked via go:linkname
					fields := strings.Fields(c.Text)
					if len(fields) == 3 {
						if m, ok := pkg.IR.Members[fields[1]]; ok {
							var obj types.Object
							switch m := m.(type) {
							case *ir.Global:
								obj = m.Object()
							case *ir.Function:
								obj = m.Object()
							default:
								panic(fmt.Sprintf("unhandled type: %T", m))
							}
							assert(obj != nil)
							g.seeAndUse(obj, nil, edgeLinkname)
						}
					}
				}
			}
		}
	}

	surroundingFunc := func(obj types.Object) *ir.Function {
		scope := obj.Parent()
		for scope != nil {
			if fn := scopes[scope]; fn != nil {
				return fn
			}
			scope = scope.Parent()
		}
		return nil
	}

	// IR form won't tell us about locally scoped types that aren't
	// being used. Walk the list of Defs to get all named types.
	//
	// IR form also won't tell us about constants; use Defs and Uses
	// to determine which constants exist and which are being used.
	for _, obj := range pkg.TypesInfo.Defs {
		switch obj := obj.(type) {
		case *types.TypeName:
			// types are being handled by walking the AST
		case *types.Const:
			g.see(obj)
			fn := surroundingFunc(obj)
			if fn == nil && obj.Exported() {
				// (1.4) packages use exported constants
				g.use(obj, nil, edgeExportedConstant)
			}
			g.typ(obj.Type(), nil)
			g.seeAndUse(obj.Type(), obj, edgeType)
		}
	}

	// Find constants being used inside functions, find sinks in tests
	for _, fn := range pkg.SrcFuncs {
		if fn.Object() != nil {
			g.see(fn.Object())
		}
		n := fn.Source()
		if n == nil {
			continue
		}
		ast.Inspect(n, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.Ident:
				obj, ok := pkg.TypesInfo.Uses[n]
				if !ok {
					return true
				}
				switch obj := obj.(type) {
				case *types.Const:
					g.seeAndUse(obj, owningObject(fn), edgeUsedConstant)
				}
			case *ast.AssignStmt:
				for _, expr := range n.Lhs {
					ident, ok := expr.(*ast.Ident)
					if !ok {
						continue
					}
					obj := pkg.TypesInfo.ObjectOf(ident)
					if obj == nil {
						continue
					}
					path := pkg.Fset.File(obj.Pos()).Name()
					if strings.HasSuffix(path, "_test.go") {
						if obj.Parent() != nil && obj.Parent().Parent() != nil && obj.Parent().Parent().Parent() == nil {
							// object's scope is the package, whose
							// parent is the file, whose parent is nil

							// (4.9) functions use package-level variables they assign to iff in tests (sinks for benchmarks)
							// (9.7) variable _reads_ use variables, writes do not, except in tests
							g.seeAndUse(obj, owningObject(fn), edgeTestSink)
						}
					}
				}
			}

			return true
		})
	}
	// Find constants being used in non-function contexts
	for _, obj := range pkg.TypesInfo.Uses {
		_, ok := obj.(*types.Const)
		if !ok {
			continue
		}
		g.seeAndUse(obj, nil, edgeUsedConstant)
	}

	var fns []*types.Func
	var fn *types.Func
	var stack []ast.Node
	for _, f := range pkg.Files {
		ast.Inspect(f, func(n ast.Node) bool {
			if n == nil {
				pop := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				if _, ok := pop.(*ast.FuncDecl); ok {
					fns = fns[:len(fns)-1]
					if len(fns) == 0 {
						fn = nil
					} else {
						fn = fns[len(fns)-1]
					}
				}
				return true
			}
			stack = append(stack, n)
			switch n := n.(type) {
			case *ast.FuncDecl:
				fn = pkg.TypesInfo.ObjectOf(n.Name).(*types.Func)
				fns = append(fns, fn)
				g.see(fn)
			case *ast.GenDecl:
				switch n.Tok {
				case token.CONST:
					groups := astutil.GroupSpecs(pkg.Fset, n.Specs)
					for _, specs := range groups {
						if len(specs) > 1 {
							cg := &constGroup{}
							g.see(cg)
							for _, spec := range specs {
								for _, name := range spec.(*ast.ValueSpec).Names {
									obj := pkg.TypesInfo.ObjectOf(name)
									// (10.1) const groups
									g.seeAndUse(obj, cg, edgeConstGroup)
									g.use(cg, obj, edgeConstGroup)
								}
							}
						}
					}
				case token.VAR:
					for _, spec := range n.Specs {
						v := spec.(*ast.ValueSpec)
						for _, name := range v.Names {
							T := pkg.TypesInfo.TypeOf(name)
							if fn != nil {
								g.seeAndUse(T, fn, edgeVarDecl)
							} else {
								// TODO(dh): we likely want to make
								// the type used by the variable, not
								// the package containing the
								// variable. But then we have to take
								// special care of blank identifiers.
								g.seeAndUse(T, nil, edgeVarDecl)
							}
							g.typ(T, nil)
						}
					}
				case token.TYPE:
					for _, spec := range n.Specs {
						// go/types doesn't provide a way to go from a
						// types.Named to the named type it was based on
						// (the t1 in type t2 t1). Therefore we walk the
						// AST and process GenDecls.
						//
						// (2.2) named types use the type they're based on
						v := spec.(*ast.TypeSpec)
						T := pkg.TypesInfo.TypeOf(v.Type)
						obj := pkg.TypesInfo.ObjectOf(v.Name)
						g.see(obj)
						g.see(T)
						g.use(T, obj, edgeType)
						g.typ(obj.Type(), nil)
						g.typ(T, nil)

						if v.Assign != 0 {
							aliasFor := obj.(*types.TypeName).Type()
							// (2.3) named types use all their aliases. we can't easily track uses of aliases
							if isIrrelevant(aliasFor) {
								// We do not track the type this is an
								// alias for (for example builtins), so
								// just mark the alias used.
								//
								// FIXME(dh): what about aliases declared inside functions?
								g.use(obj, nil, edgeAlias)
							} else {
								g.see(aliasFor)
								g.seeAndUse(obj, aliasFor, edgeAlias)
							}
						}
					}
				}
			}
			return true
		})
	}

	for _, m := range pkg.IR.Members {
		switch m := m.(type) {
		case *ir.NamedConst:
			// nothing to do, we collect all constants from Defs
		case *ir.Global:
			if m.Object() != nil {
				g.see(m.Object())
				if m.Object().Exported() {
					// (1.3) packages use exported variables
					g.use(m.Object(), nil, edgeExportedVariable)
				}
			}
		case *ir.Function:
			mObj := owningObject(m)
			if mObj != nil {
				g.see(mObj)
			}
			//lint:ignore SA9003 handled implicitly
			if m.Name() == "init" {
				// (1.5) packages use init functions
				//
				// This is handled implicitly. The generated init
				// function has no object, thus everything in it will
				// be owned by the package.
			}
			// This branch catches top-level functions, not methods.
			if m.Object() != nil && m.Object().Exported() {
				// (1.2) packages use exported functions
				g.use(mObj, nil, edgeExportedFunction)
			}
			if m.Name() == "main" && pkg.Pkg.Name() == "main" {
				// (1.7) packages use the main function iff in the main package
				g.use(mObj, nil, edgeMainFunction)
			}
			if pkg.Pkg.Path() == "runtime" && runtimeFuncs[m.Name()] {
				// (9.8) runtime functions that may be called from user code via the compiler
				g.use(mObj, nil, edgeRuntimeFunction)
			}
			if m.Source() != nil {
				doc := m.Source().(*ast.FuncDecl).Doc
				if doc != nil {
					for _, cmt := range doc.List {
						if strings.HasPrefix(cmt.Text, "//go:cgo_export_") {
							// (1.6) packages use functions exported to cgo
							g.use(mObj, nil, edgeCgoExported)
						}
					}
				}
			}
			g.function(m)
		case *ir.Type:
			g.see(m.Object())
			if m.Object().Exported() {
				// (1.1) packages use exported named types
				g.use(m.Object(), nil, edgeExportedType)
			}
			g.typ(m.Type(), nil)
		default:
			panic(fmt.Sprintf("unreachable: %T", m))
		}
	}

	// OPT(dh): can we find meaningful initial capacities for these slices?
	var ifaces []*types.Interface
	var notIfaces []types.Type

	g.seenTypes.Iterate(func(t types.Type, _ interface{}) {
		switch t := t.(type) {
		case *types.Interface:
			// OPT(dh): (8.1) we only need interfaces that have unexported methods
			ifaces = append(ifaces, t)
		default:
			if _, ok := t.Underlying().(*types.Interface); !ok {
				notIfaces = append(notIfaces, t)
			}
		}
	})

	// (8.0) handle interfaces
	for _, t := range notIfaces {
		ms := pkg.IR.Prog.MethodSets.MethodSet(t)
		for _, iface := range ifaces {
			if sels, ok := g.implements(t, iface, ms); ok {
				for _, sel := range sels {
					g.useMethod(t, sel, t, edgeImplements)
				}
			}
		}
	}

	type ignoredKey struct {
		file string
		line int
	}
	ignores := map[ignoredKey]struct{}{}
	for _, dir := range pkg.Directives {
		if dir.Command != "ignore" && dir.Command != "file-ignore" {
			continue
		}
		if len(dir.Arguments) == 0 {
			continue
		}
		for _, check := range strings.Split(dir.Arguments[0], ",") {
			if check == "U1000" {
				pos := pkg.Fset.PositionFor(dir.Node.Pos(), false)
				var key ignoredKey
				switch dir.Command {
				case "ignore":
					key = ignoredKey{
						pos.Filename,
						pos.Line,
					}
				case "file-ignore":
					key = ignoredKey{
						pos.Filename,
						-1,
					}
				}

				ignores[key] = struct{}{}
				break
			}
		}
	}

	if len(ignores) > 0 {
		// all objects annotated with a //lint:ignore U1000 are considered used
		for obj := range g.Nodes {
			if obj, ok := obj.(types.Object); ok {
				pos := pkg.Fset.PositionFor(obj.Pos(), false)
				key1 := ignoredKey{
					pos.Filename,
					pos.Line,
				}
				key2 := ignoredKey{
					pos.Filename,
					-1,
				}
				_, ok := ignores[key1]
				if !ok {
					_, ok = ignores[key2]
				}
				if ok {
					g.use(obj, nil, edgeIgnored)

					// use methods and fields of ignored types
					if obj, ok := obj.(*types.TypeName); ok {
						if obj.IsAlias() {
							if typ, ok := obj.Type().(*types.Named); ok && typ.Obj().Pkg() != obj.Pkg() {
								// This is an alias of a named type in another package.
								// Don't walk its fields or methods; we don't have to,
								// and it breaks an assertion in graph.use because we're using an object that we haven't seen before.
								//
								// For aliases to types in the same package, we do want to ignore the fields and methods,
								// because ignoring the alias should ignore the aliased type.
								continue
							}
						}
						if typ, ok := obj.Type().(*types.Named); ok {
							for i := 0; i < typ.NumMethods(); i++ {
								g.use(typ.Method(i), nil, edgeIgnored)
							}
						}
						if typ, ok := obj.Type().Underlying().(*types.Struct); ok {
							for i := 0; i < typ.NumFields(); i++ {
								g.use(typ.Field(i), nil, edgeIgnored)
							}
						}
					}
				}
			}
		}
	}
}

func (g *graph) useMethod(t types.Type, sel *types.Selection, by interface{}, kind edgeKind) {
	obj := sel.Obj()
	path := sel.Index()
	assert(obj != nil)
	if len(path) > 1 {
		base := typeutil.Dereference(t).Underlying().(*types.Struct)
		for _, idx := range path[:len(path)-1] {
			next := base.Field(idx)
			// (6.3) structs use embedded fields that help implement interfaces
			g.see(base)
			g.seeAndUse(next, base, edgeProvidesMethod)
			base, _ = typeutil.Dereference(next.Type()).Underlying().(*types.Struct)
		}
	}
	g.seeAndUse(obj, by, kind)
}

func owningObject(fn *ir.Function) types.Object {
	if fn.Object() != nil {
		return fn.Object()
	}
	if fn.Parent() != nil {
		return owningObject(fn.Parent())
	}
	return nil
}

func (g *graph) function(fn *ir.Function) {
	if fn.Package() != nil && fn.Package() != g.pkg.IR {
		return
	}

	if _, ok := g.seenFns[fn]; ok {
		return
	}
	g.seenFns[fn] = struct{}{}

	// (4.1) functions use all their arguments, return parameters and receivers
	g.signature(fn.Signature, owningObject(fn))
	g.instructions(fn)
	for _, anon := range fn.AnonFuncs {
		// (4.2) functions use anonymous functions defined beneath them
		//
		// This fact is expressed implicitly. Anonymous functions have
		// no types.Object, so their owner is the surrounding
		// function.
		g.function(anon)
	}
}

func (g *graph) typ(t types.Type, parent types.Type) {
	if g.seenTypes.At(t) != nil {
		return
	}

	if t, ok := t.(*types.Named); ok && t.Obj().Pkg() != nil {
		if t.Obj().Pkg() != g.pkg.Pkg {
			return
		}
	}

	g.seenTypes.Set(t, struct{}{})
	if isIrrelevant(t) {
		return
	}

	g.see(t)
	switch t := t.(type) {
	case *types.Struct:
		for i := 0; i < t.NumFields(); i++ {
			g.see(t.Field(i))
			if t.Field(i).Exported() {
				// (6.2) structs use exported fields
				g.use(t.Field(i), t, edgeExportedField)
			} else if t.Field(i).Name() == "_" {
				g.use(t.Field(i), t, edgeBlankField)
			} else if isNoCopyType(t.Field(i).Type()) {
				// (6.1) structs use fields of type NoCopy sentinel
				g.use(t.Field(i), t, edgeNoCopySentinel)
			} else if parent == nil {
				// (11.1) anonymous struct types use all their fields.
				g.use(t.Field(i), t, edgeAnonymousStruct)
			}
			if t.Field(i).Anonymous() {
				// does the embedded field contribute exported methods to the method set?
				T := t.Field(i).Type()
				if _, ok := T.Underlying().(*types.Pointer); !ok {
					// An embedded field is addressable, so check
					// the pointer type to get the full method set
					T = types.NewPointer(T)
				}
				ms := g.pkg.IR.Prog.MethodSets.MethodSet(T)
				for j := 0; j < ms.Len(); j++ {
					if ms.At(j).Obj().Exported() {
						// (6.4) structs use embedded fields that have exported methods (recursively)
						g.use(t.Field(i), t, edgeExtendsExportedMethodSet)
						break
					}
				}

				seen := map[*types.Struct]struct{}{}
				var hasExportedField func(t types.Type) bool
				hasExportedField = func(T types.Type) bool {
					t, ok := typeutil.Dereference(T).Underlying().(*types.Struct)
					if !ok {
						return false
					}
					if _, ok := seen[t]; ok {
						return false
					}
					seen[t] = struct{}{}
					for i := 0; i < t.NumFields(); i++ {
						field := t.Field(i)
						if field.Exported() {
							return true
						}
						if field.Embedded() && hasExportedField(field.Type()) {
							return true
						}
					}
					return false
				}
				// does the embedded field contribute exported fields?
				if hasExportedField(t.Field(i).Type()) {
					// (6.5) structs use embedded structs that have exported fields (recursively)
					g.use(t.Field(i), t, edgeExtendsExportedFields)
				}

			}
			g.variable(t.Field(i))
		}
	case *types.Basic:
		// Nothing to do
	case *types.Named:
		// (9.3) types use their underlying and element types
		g.seeAndUse(t.Underlying(), t, edgeUnderlyingType)
		g.seeAndUse(t.Obj(), t, edgeTypeName)
		g.seeAndUse(t, t.Obj(), edgeNamedType)

		// (2.4) named types use the pointer type
		if _, ok := t.Underlying().(*types.Interface); !ok && t.NumMethods() > 0 {
			g.seeAndUse(types.NewPointer(t), t, edgePointerType)
		}

		for i := 0; i < t.NumMethods(); i++ {
			g.see(t.Method(i))
			// don't use trackExportedIdentifier here, we care about
			// all exported methods, even in package main or in tests.
			if t.Method(i).Exported() {
				// (2.1) named types use exported methods
				g.use(t.Method(i), t, edgeExportedMethod)
			}
			g.function(g.pkg.IR.Prog.FuncValue(t.Method(i)))
		}

		g.typ(t.Underlying(), t)
	case *types.Slice:
		// (9.3) types use their underlying and element types
		g.seeAndUse(t.Elem(), t, edgeElementType)
		g.typ(t.Elem(), nil)
	case *types.Map:
		// (9.3) types use their underlying and element types
		g.seeAndUse(t.Elem(), t, edgeElementType)
		// (9.3) types use their underlying and element types
		g.seeAndUse(t.Key(), t, edgeKeyType)
		g.typ(t.Elem(), nil)
		g.typ(t.Key(), nil)
	case *types.Signature:
		g.signature(t, nil)
	case *types.Interface:
		for i := 0; i < t.NumMethods(); i++ {
			m := t.Method(i)
			// (8.3) All interface methods are marked as used
			g.seeAndUse(m, t, edgeInterfaceMethod)
			g.seeAndUse(m.Type().(*types.Signature), m, edgeSignature)
			g.signature(m.Type().(*types.Signature), nil)
		}
		for i := 0; i < t.NumEmbeddeds(); i++ {
			tt := t.EmbeddedType(i)
			// (8.4) All embedded interfaces are marked as used
			g.seeAndUse(tt, t, edgeEmbeddedInterface)
		}
	case *types.Array:
		// (9.3) types use their underlying and element types
		g.seeAndUse(t.Elem(), t, edgeElementType)
		g.typ(t.Elem(), nil)
	case *types.Pointer:
		// (9.3) types use their underlying and element types
		g.seeAndUse(t.Elem(), t, edgeElementType)
		g.typ(t.Elem(), nil)
	case *types.Chan:
		// (9.3) types use their underlying and element types
		g.seeAndUse(t.Elem(), t, edgeElementType)
		g.typ(t.Elem(), nil)
	case *types.Tuple:
		for i := 0; i < t.Len(); i++ {
			// (9.3) types use their underlying and element types
			g.seeAndUse(t.At(i).Type(), t, edgeTupleElement|edgeType)
			g.typ(t.At(i).Type(), nil)
		}
	default:
		panic(fmt.Sprintf("unreachable: %T", t))
	}
}

func (g *graph) variable(v *types.Var) {
	// (9.2) variables use their types
	g.seeAndUse(v.Type(), v, edgeType)
	g.typ(v.Type(), nil)
}

func (g *graph) signature(sig *types.Signature, fn types.Object) {
	var user interface{} = fn
	if fn == nil {
		user = sig
		g.see(sig)
	}
	if sig.Recv() != nil {
		g.seeAndUse(sig.Recv().Type(), user, edgeReceiver|edgeType)
		g.typ(sig.Recv().Type(), nil)
	}
	for i := 0; i < sig.Params().Len(); i++ {
		param := sig.Params().At(i)
		g.seeAndUse(param.Type(), user, edgeFunctionArgument|edgeType)
		g.typ(param.Type(), nil)
	}
	for i := 0; i < sig.Results().Len(); i++ {
		param := sig.Results().At(i)
		g.seeAndUse(param.Type(), user, edgeFunctionResult|edgeType)
		g.typ(param.Type(), nil)
	}
}

func (g *graph) instructions(fn *ir.Function) {
	fnObj := owningObject(fn)
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			ops := instr.Operands(nil)
			switch instr.(type) {
			case *ir.Store:
				// (9.7) variable _reads_ use variables, writes do not
				ops = ops[1:]
			case *ir.DebugRef:
				ops = nil
			}
			for _, arg := range ops {
				walkPhi(*arg, func(v ir.Value) {
					switch v := v.(type) {
					case *ir.Function:
						// (4.3) functions use closures and bound methods.
						// (4.5) functions use functions they call
						// (9.5) instructions use their operands
						// (4.4) functions use functions they return. we assume that someone else will call the returned function
						if owningObject(v) != nil {
							g.seeAndUse(owningObject(v), fnObj, edgeInstructionOperand)
						}
						g.function(v)
					case *ir.Const:
						// (9.6) instructions use their operands' types
						g.seeAndUse(v.Type(), fnObj, edgeType)
						g.typ(v.Type(), nil)
					case *ir.Global:
						if v.Object() != nil {
							// (9.5) instructions use their operands
							g.seeAndUse(v.Object(), fnObj, edgeInstructionOperand)
						}
					}
				})
			}
			if v, ok := instr.(ir.Value); ok {
				if _, ok := v.(*ir.Range); !ok {
					// See https://github.com/golang/go/issues/19670

					// (4.8) instructions use their types
					// (9.4) conversions use the type they convert to
					g.seeAndUse(v.Type(), fnObj, edgeType)
					g.typ(v.Type(), nil)
				}
			}
			switch instr := instr.(type) {
			case *ir.Field:
				st := instr.X.Type().Underlying().(*types.Struct)
				field := st.Field(instr.Field)
				// (4.7) functions use fields they access
				g.seeAndUse(field, fnObj, edgeFieldAccess)
			case *ir.FieldAddr:
				st := typeutil.Dereference(instr.X.Type()).Underlying().(*types.Struct)
				field := st.Field(instr.Field)
				// (4.7) functions use fields they access
				g.seeAndUse(field, fnObj, edgeFieldAccess)
			case *ir.Store:
				// nothing to do, handled generically by operands
			case *ir.Call:
				c := instr.Common()
				if !c.IsInvoke() {
					// handled generically as an instruction operand
				} else {
					// (4.5) functions use functions/interface methods they call
					g.seeAndUse(c.Method, fnObj, edgeInterfaceCall)
				}
			case *ir.Return:
				// nothing to do, handled generically by operands
			case *ir.ChangeType:
				// conversion type handled generically

				s1, ok1 := typeutil.Dereference(instr.Type()).Underlying().(*types.Struct)
				s2, ok2 := typeutil.Dereference(instr.X.Type()).Underlying().(*types.Struct)
				if ok1 && ok2 {
					// Converting between two structs. The fields are
					// relevant for the conversion, but only if the
					// fields are also used outside of the conversion.
					// Mark fields as used by each other.

					assert(s1.NumFields() == s2.NumFields())
					for i := 0; i < s1.NumFields(); i++ {
						g.see(s1.Field(i))
						g.see(s2.Field(i))
						// (5.1) when converting between two equivalent structs, the fields in
						// either struct use each other. the fields are relevant for the
						// conversion, but only if the fields are also accessed outside the
						// conversion.
						g.seeAndUse(s1.Field(i), s2.Field(i), edgeStructConversion)
						g.seeAndUse(s2.Field(i), s1.Field(i), edgeStructConversion)
					}
				}
			case *ir.MakeInterface:
				// nothing to do, handled generically by operands
			case *ir.Slice:
				// nothing to do, handled generically by operands
			case *ir.RunDefers:
				// nothing to do, the deferred functions are already marked use by deferring them.
			case *ir.Convert:
				// to unsafe.Pointer
				if typ, ok := instr.Type().(*types.Basic); ok && typ.Kind() == types.UnsafePointer {
					if ptr, ok := instr.X.Type().Underlying().(*types.Pointer); ok {
						if st, ok := ptr.Elem().Underlying().(*types.Struct); ok {
							for i := 0; i < st.NumFields(); i++ {
								// (5.2) when converting to or from unsafe.Pointer, mark all fields as used.
								g.seeAndUse(st.Field(i), fnObj, edgeUnsafeConversion)
							}
						}
					}
				}
				// from unsafe.Pointer
				if typ, ok := instr.X.Type().(*types.Basic); ok && typ.Kind() == types.UnsafePointer {
					if ptr, ok := instr.Type().Underlying().(*types.Pointer); ok {
						if st, ok := ptr.Elem().Underlying().(*types.Struct); ok {
							for i := 0; i < st.NumFields(); i++ {
								// (5.2) when converting to or from unsafe.Pointer, mark all fields as used.
								g.seeAndUse(st.Field(i), fnObj, edgeUnsafeConversion)
							}
						}
					}
				}
			case *ir.TypeAssert:
				// nothing to do, handled generically by instruction
				// type (possibly a tuple, which contains the asserted
				// to type). redundantly handled by the type of
				// ir.Extract, too
			case *ir.MakeClosure:
				// nothing to do, handled generically by operands
			case *ir.Alloc:
				// nothing to do
			case *ir.UnOp:
				// nothing to do
			case *ir.BinOp:
				// nothing to do
			case *ir.If:
				// nothing to do
			case *ir.Jump:
				// nothing to do
			case *ir.Unreachable:
				// nothing to do
			case *ir.IndexAddr:
				// nothing to do
			case *ir.Extract:
				// nothing to do
			case *ir.Panic:
				// nothing to do
			case *ir.DebugRef:
				// nothing to do
			case *ir.BlankStore:
				// nothing to do
			case *ir.Phi:
				// nothing to do
			case *ir.Sigma:
				// nothing to do
			case *ir.MakeMap:
				// nothing to do
			case *ir.MapUpdate:
				// nothing to do
			case *ir.MapLookup:
				// nothing to do
			case *ir.StringLookup:
				// nothing to do
			case *ir.MakeSlice:
				// nothing to do
			case *ir.Send:
				// nothing to do
			case *ir.MakeChan:
				// nothing to do
			case *ir.Range:
				// nothing to do
			case *ir.Next:
				// nothing to do
			case *ir.Index:
				// nothing to do
			case *ir.Select:
				// nothing to do
			case *ir.ChangeInterface:
				// nothing to do
			case *ir.Load:
				// nothing to do
			case *ir.Go:
				// nothing to do
			case *ir.Defer:
				// nothing to do
			case *ir.Parameter:
				// nothing to do
			case *ir.Const:
				// nothing to do
			case *ir.Recv:
				// nothing to do
			case *ir.TypeSwitch:
				// nothing to do
			case *ir.ConstantSwitch:
				// nothing to do
			case *ir.SliceToArrayPointer:
				// nothing to do
			default:
				lint.ExhaustiveTypeSwitch(instr)
			}
		}
	}
}

// isNoCopyType reports whether a type represents the NoCopy sentinel
// type. The NoCopy type is a named struct with no fields and exactly
// one method `func Lock()` that is empty.
//
// FIXME(dh): currently we're not checking that the function body is
// empty.
func isNoCopyType(typ types.Type) bool {
	st, ok := typ.Underlying().(*types.Struct)
	if !ok {
		return false
	}
	if st.NumFields() != 0 {
		return false
	}

	named, ok := typ.(*types.Named)
	if !ok {
		return false
	}
	if named.NumMethods() != 1 {
		return false
	}
	meth := named.Method(0)
	if meth.Name() != "Lock" {
		return false
	}
	sig := meth.Type().(*types.Signature)
	if sig.Params().Len() != 0 || sig.Results().Len() != 0 {
		return false
	}
	return true
}

func walkPhi(v ir.Value, fn func(v ir.Value)) {
	phi, ok := v.(*ir.Phi)
	if !ok {
		fn(v)
		return
	}

	seen := map[ir.Value]struct{}{}
	var impl func(v *ir.Phi)
	impl = func(v *ir.Phi) {
		if _, ok := seen[v]; ok {
			return
		}
		seen[v] = struct{}{}
		for _, e := range v.Edges {
			if ev, ok := e.(*ir.Phi); ok {
				impl(ev)
			} else {
				fn(e)
			}
		}
	}
	impl(phi)
}
