package vrp

// TODO(dh) widening and narrowing have a lot of code in common. Make
// it reusable.

import (
	"fmt"
	"go/constant"
	"go/token"
	"go/types"
	"math/big"
	"sort"
	"strings"

	"honnef.co/go/tools/lint"
	"honnef.co/go/tools/ssa"
)

type Future interface {
	Constraint
	Futures() []ssa.Value
	Resolve()
	IsKnown() bool
	MarkUnresolved()
	MarkResolved()
	IsResolved() bool
}

type Range interface {
	Union(other Range) Range
	IsKnown() bool
}

type Constraint interface {
	Y() ssa.Value
	isConstraint()
	String() string
	Eval(*Graph) Range
	Operands() []ssa.Value
}

type aConstraint struct {
	y ssa.Value
}

func NewConstraint(y ssa.Value) aConstraint {
	return aConstraint{y}
}

func (aConstraint) isConstraint()  {}
func (c aConstraint) Y() ssa.Value { return c.y }

type PhiConstraint struct {
	aConstraint
	Vars []ssa.Value
}

func NewPhiConstraint(vars []ssa.Value, y ssa.Value) Constraint {
	uniqm := map[ssa.Value]struct{}{}
	for _, v := range vars {
		uniqm[v] = struct{}{}
	}
	var uniq []ssa.Value
	for v := range uniqm {
		uniq = append(uniq, v)
	}
	return &PhiConstraint{
		aConstraint: NewConstraint(y),
		Vars:        uniq,
	}
}

func (c *PhiConstraint) Operands() []ssa.Value {
	return c.Vars
}

func (c *PhiConstraint) Eval(g *Graph) Range {
	i := Range(nil)
	for _, v := range c.Vars {
		i = g.Range(v).Union(i)
	}
	return i
}

func (c *PhiConstraint) String() string {
	names := make([]string, len(c.Vars))
	for i, v := range c.Vars {
		names[i] = v.Name()
	}
	return fmt.Sprintf("%s = φ(%s)", c.Y().Name(), strings.Join(names, ", "))
}

func isSupportedType(typ types.Type) bool {
	switch typ := typ.Underlying().(type) {
	case *types.Basic:
		switch typ.Kind() {
		case types.String, types.UntypedString:
			return true
		default:
			if (typ.Info() & types.IsInteger) == 0 {
				return false
			}
		}
	case *types.Chan:
		return true
	case *types.Slice:
		return true
	default:
		return false
	}
	return true
}

func ConstantToZ(c constant.Value) Z {
	s := constant.ToInt(c).ExactString()
	n := &big.Int{}
	n.SetString(s, 10)
	return NewBigZ(n)
}

func sigmaInteger(g *Graph, ins *ssa.Sigma, cond *ssa.BinOp, ops []*ssa.Value) Constraint {
	op := cond.Op
	if !ins.Branch {
		op = (invertToken(op))
	}

	switch op {
	case token.EQL, token.GTR, token.GEQ, token.LSS, token.LEQ:
	default:
		return nil
	}
	var a, b ssa.Value
	if (*ops[0]) == ins.X {
		a = *ops[0]
		b = *ops[1]
	} else {
		a = *ops[1]
		b = *ops[0]
		op = flipToken(op)
	}
	return NewIntIntersectionConstraint(a, b, op, g.ranges, ins)
}

func sigmaString(g *Graph, ins *ssa.Sigma, cond *ssa.BinOp, ops []*ssa.Value) Constraint {
	op := cond.Op
	if !ins.Branch {
		op = (invertToken(op))
	}

	switch op {
	case token.EQL, token.GTR, token.GEQ, token.LSS, token.LEQ:
	default:
		return nil
	}

	if ((*ops[0]).Type().Underlying().(*types.Basic).Info() & types.IsString) == 0 {
		var a, b ssa.Value
		call, ok := (*ops[0]).(*ssa.Call)
		if ok && call.Common().Args[0] == ins.X {
			a = *ops[0]
			b = *ops[1]
		} else {
			a = *ops[1]
			b = *ops[0]
			op = flipToken(op)
		}
		return NewStringIntersectionConstraint(a, b, op, g.ranges, ins)
	}
	var a, b ssa.Value
	if (*ops[0]) == ins.X {
		a = *ops[0]
		b = *ops[1]
	} else {
		a = *ops[1]
		b = *ops[0]
		op = flipToken(op)
	}
	return NewStringIntersectionConstraint(a, b, op, g.ranges, ins)
}

func sigmaSlice(g *Graph, ins *ssa.Sigma, cond *ssa.BinOp, ops []*ssa.Value) Constraint {
	// TODO(dh) sigmaSlice and sigmaString are a lot alike. Can they
	// be merged?
	//
	// XXX support futures

	op := cond.Op
	if !ins.Branch {
		op = (invertToken(op))
	}

	k, ok := (*ops[1]).(*ssa.Const)
	// XXX investigate in what cases this wouldn't be a Const
	//
	// XXX what if left and right are swapped?
	if !ok {
		return nil
	}

	call, ok := (*ops[0]).(*ssa.Call)
	if !ok {
		return nil
	}
	builtin, ok := call.Common().Value.(*ssa.Builtin)
	if !ok {
		return nil
	}
	if builtin.Name() != "len" {
		return nil
	}
	callops := call.Operands(nil)

	v := ConstantToZ(k.Value)
	c := NewSliceIntersectionConstraint(*callops[1], IntInterval{}, ins).(*SliceIntersectionConstraint)
	switch op {
	case token.EQL:
		c.I = NewIntInterval(v, v)
	case token.GTR, token.GEQ:
		off := int64(0)
		if cond.Op == token.GTR {
			off = 1
		}
		c.I = NewIntInterval(
			v.Add(NewZ(off)),
			PInfinity,
		)
	case token.LSS, token.LEQ:
		off := int64(0)
		if cond.Op == token.LSS {
			off = -1
		}
		c.I = NewIntInterval(
			NInfinity,
			v.Add(NewZ(off)),
		)
	default:
		return nil
	}
	return c
}

func BuildGraph(f *ssa.Function) *Graph {
	g := &Graph{
		Vertices: map[interface{}]*Vertex{},
		ranges:   Ranges{},
	}

	var cs []Constraint

	ops := make([]*ssa.Value, 16)
	seen := map[ssa.Value]bool{}
	for _, block := range f.Blocks {
		for _, ins := range block.Instrs {
			ops = ins.Operands(ops[:0])
			for _, op := range ops {
				if c, ok := (*op).(*ssa.Const); ok {
					if seen[c] {
						continue
					}
					seen[c] = true
					if c.Value == nil {
						switch c.Type().Underlying().(type) {
						case *types.Slice:
							cs = append(cs, NewSliceIntervalConstraint(NewIntInterval(NewZ(0), NewZ(0)), c))
						}
						continue
					}
					switch c.Value.Kind() {
					case constant.Int:
						v := ConstantToZ(c.Value)
						cs = append(cs, NewIntIntervalConstraint(NewIntInterval(v, v), c))
					case constant.String:
						s := constant.StringVal(c.Value)
						n := NewZ(int64(len(s)))
						cs = append(cs, NewStringIntervalConstraint(NewIntInterval(n, n), c))
					}
				}
			}
		}
	}
	for _, block := range f.Blocks {
		for _, ins := range block.Instrs {
			switch ins := ins.(type) {
			case *ssa.Convert:
				switch v := ins.Type().Underlying().(type) {
				case *types.Basic:
					if (v.Info() & types.IsInteger) == 0 {
						continue
					}
					cs = append(cs, NewIntConversionConstraint(ins.X, ins))
				}
			case *ssa.Call:
				if static := ins.Common().StaticCallee(); static != nil {
					if fn, ok := static.Object().(*types.Func); ok {
						switch lint.FuncName(fn) {
						case "bytes.Index", "bytes.IndexAny", "bytes.IndexByte",
							"bytes.IndexFunc", "bytes.IndexRune", "bytes.LastIndex",
							"bytes.LastIndexAny", "bytes.LastIndexByte", "bytes.LastIndexFunc",
							"strings.Index", "strings.IndexAny", "strings.IndexByte",
							"strings.IndexFunc", "strings.IndexRune", "strings.LastIndex",
							"strings.LastIndexAny", "strings.LastIndexByte", "strings.LastIndexFunc":
							// TODO(dh): instead of limiting by +∞,
							// limit by the upper bound of the passed
							// string
							cs = append(cs, NewIntIntervalConstraint(NewIntInterval(NewZ(-1), PInfinity), ins))
						case "bytes.Title", "bytes.ToLower", "bytes.ToTitle", "bytes.ToUpper",
							"strings.Title", "strings.ToLower", "strings.ToTitle", "strings.ToUpper":
							cs = append(cs, NewCopyConstraint(ins.Common().Args[0], ins))
						case "bytes.ToLowerSpecial", "bytes.ToTitleSpecial", "bytes.ToUpperSpecial",
							"strings.ToLowerSpecial", "strings.ToTitleSpecial", "strings.ToUpperSpecial":
							cs = append(cs, NewCopyConstraint(ins.Common().Args[1], ins))
						case "bytes.Compare", "strings.Compare":
							cs = append(cs, NewIntIntervalConstraint(NewIntInterval(NewZ(-1), NewZ(1)), ins))
						case "bytes.Count", "strings.Count":
							// TODO(dh): instead of limiting by +∞,
							// limit by the upper bound of the passed
							// string.
							cs = append(cs, NewIntIntervalConstraint(NewIntInterval(NewZ(0), PInfinity), ins))
						case "bytes.Map", "bytes.TrimFunc", "bytes.TrimLeft", "bytes.TrimLeftFunc",
							"bytes.TrimRight", "bytes.TrimRightFunc", "bytes.TrimSpace",
							"strings.Map", "strings.TrimFunc", "strings.TrimLeft", "strings.TrimLeftFunc",
							"strings.TrimRight", "strings.TrimRightFunc", "strings.TrimSpace":
							// TODO(dh): lower = 0, upper = upper of passed string
						case "bytes.TrimPrefix", "bytes.TrimSuffix",
							"strings.TrimPrefix", "strings.TrimSuffix":
							// TODO(dh) range between "unmodified" and len(cutset) removed
						case "(*bytes.Buffer).Cap", "(*bytes.Buffer).Len", "(*bytes.Reader).Len", "(*bytes.Reader).Size":
							cs = append(cs, NewIntIntervalConstraint(NewIntInterval(NewZ(0), PInfinity), ins))
						}
					}
				}
				builtin, ok := ins.Common().Value.(*ssa.Builtin)
				ops := ins.Operands(nil)
				if !ok {
					continue
				}
				switch builtin.Name() {
				case "len":
					switch op1 := (*ops[1]).Type().Underlying().(type) {
					case *types.Basic:
						if op1.Kind() == types.String || op1.Kind() == types.UntypedString {
							cs = append(cs, NewStringLengthConstraint(*ops[1], ins))
						}
					case *types.Slice:
						cs = append(cs, NewSliceLengthConstraint(*ops[1], ins))
					}

				case "append":
					cs = append(cs, NewSliceAppendConstraint(ins.Common().Args[0], ins.Common().Args[1], ins))
				}
			case *ssa.BinOp:
				ops := ins.Operands(nil)
				basic, ok := (*ops[0]).Type().Underlying().(*types.Basic)
				if !ok {
					continue
				}
				switch basic.Kind() {
				case types.Int, types.Int8, types.Int16, types.Int32, types.Int64,
					types.Uint, types.Uint8, types.Uint16, types.Uint32, types.Uint64, types.UntypedInt:
					fns := map[token.Token]func(ssa.Value, ssa.Value, ssa.Value) Constraint{
						token.ADD: NewIntAddConstraint,
						token.SUB: NewIntSubConstraint,
						token.MUL: NewIntMulConstraint,
						// XXX support QUO, REM, SHL, SHR
					}
					fn, ok := fns[ins.Op]
					if ok {
						cs = append(cs, fn(*ops[0], *ops[1], ins))
					}
				case types.String, types.UntypedString:
					if ins.Op == token.ADD {
						cs = append(cs, NewStringConcatConstraint(*ops[0], *ops[1], ins))
					}
				}
			case *ssa.Slice:
				typ := ins.X.Type().Underlying()
				switch typ := typ.(type) {
				case *types.Basic:
					cs = append(cs, NewStringSliceConstraint(ins.X, ins.Low, ins.High, ins))
				case *types.Slice:
					cs = append(cs, NewSliceSliceConstraint(ins.X, ins.Low, ins.High, ins))
				case *types.Array:
					cs = append(cs, NewArraySliceConstraint(ins.X, ins.Low, ins.High, ins))
				case *types.Pointer:
					if _, ok := typ.Elem().(*types.Array); !ok {
						continue
					}
					cs = append(cs, NewArraySliceConstraint(ins.X, ins.Low, ins.High, ins))
				}
			case *ssa.Phi:
				if !isSupportedType(ins.Type()) {
					continue
				}
				ops := ins.Operands(nil)
				dops := make([]ssa.Value, len(ops))
				for i, op := range ops {
					dops[i] = *op
				}
				cs = append(cs, NewPhiConstraint(dops, ins))
			case *ssa.Sigma:
				pred := ins.Block().Preds[0]
				instrs := pred.Instrs
				cond, ok := instrs[len(instrs)-1].(*ssa.If).Cond.(*ssa.BinOp)
				ops := cond.Operands(nil)
				if !ok {
					continue
				}
				switch typ := ins.Type().Underlying().(type) {
				case *types.Basic:
					var c Constraint
					switch typ.Kind() {
					case types.Int, types.Int8, types.Int16, types.Int32, types.Int64,
						types.Uint, types.Uint8, types.Uint16, types.Uint32, types.Uint64, types.UntypedInt:
						c = sigmaInteger(g, ins, cond, ops)
					case types.String, types.UntypedString:
						c = sigmaString(g, ins, cond, ops)
					}
					if c != nil {
						cs = append(cs, c)
					}
				case *types.Slice:
					c := sigmaSlice(g, ins, cond, ops)
					if c != nil {
						cs = append(cs, c)
					}
				default:
					//log.Printf("unsupported sigma type %T", typ) // XXX
				}
			case *ssa.MakeChan:
				cs = append(cs, NewMakeChannelConstraint(ins.Size, ins))
			case *ssa.MakeSlice:
				cs = append(cs, NewMakeSliceConstraint(ins.Len, ins))
			case *ssa.ChangeType:
				switch ins.X.Type().Underlying().(type) {
				case *types.Chan:
					cs = append(cs, NewChannelChangeTypeConstraint(ins.X, ins))
				}
			}
		}
	}

	for _, c := range cs {
		if c == nil {
			panic("nil constraint")
		}
		// If V is used in constraint C, then we create an edge V->C
		for _, op := range c.Operands() {
			g.AddEdge(op, c, false)
		}
		if c, ok := c.(Future); ok {
			for _, op := range c.Futures() {
				g.AddEdge(op, c, true)
			}
		}
		// If constraint C defines variable V, then we create an edge
		// C->V
		g.AddEdge(c, c.Y(), false)
	}

	g.FindSCCs()
	g.sccEdges = make([][]Edge, len(g.SCCs))
	g.futures = make([][]Future, len(g.SCCs))
	for _, e := range g.Edges {
		g.sccEdges[e.From.SCC] = append(g.sccEdges[e.From.SCC], e)
		if !e.control {
			continue
		}
		if c, ok := e.To.Value.(Future); ok {
			g.futures[e.From.SCC] = append(g.futures[e.From.SCC], c)
		}
	}
	return g
}

func (g *Graph) Solve() Ranges {
	var consts []Z
	off := NewZ(1)
	for _, n := range g.Vertices {
		if c, ok := n.Value.(*ssa.Const); ok {
			basic, ok := c.Type().Underlying().(*types.Basic)
			if !ok {
				continue
			}
			if (basic.Info() & types.IsInteger) != 0 {
				z := ConstantToZ(c.Value)
				consts = append(consts, z)
				consts = append(consts, z.Add(off))
				consts = append(consts, z.Sub(off))
			}
		}

	}
	sort.Sort(Zs(consts))

	for scc, vertices := range g.SCCs {
		n := 0
		n = len(vertices)
		if n == 1 {
			g.resolveFutures(scc)
			v := vertices[0]
			if v, ok := v.Value.(ssa.Value); ok {
				switch typ := v.Type().Underlying().(type) {
				case *types.Basic:
					switch typ.Kind() {
					case types.String, types.UntypedString:
						if !g.Range(v).(StringInterval).IsKnown() {
							g.SetRange(v, StringInterval{NewIntInterval(NewZ(0), PInfinity)})
						}
					default:
						if !g.Range(v).(IntInterval).IsKnown() {
							g.SetRange(v, InfinityFor(v))
						}
					}
				case *types.Chan:
					if !g.Range(v).(ChannelInterval).IsKnown() {
						g.SetRange(v, ChannelInterval{NewIntInterval(NewZ(0), PInfinity)})
					}
				case *types.Slice:
					if !g.Range(v).(SliceInterval).IsKnown() {
						g.SetRange(v, SliceInterval{NewIntInterval(NewZ(0), PInfinity)})
					}
				}
			}
			if c, ok := v.Value.(Constraint); ok {
				g.SetRange(c.Y(), c.Eval(g))
			}
		} else {
			uses := g.uses(scc)
			entries := g.entries(scc)
			for len(entries) > 0 {
				v := entries[len(entries)-1]
				entries = entries[:len(entries)-1]
				for _, use := range uses[v] {
					if g.widen(use, consts) {
						entries = append(entries, use.Y())
					}
				}
			}

			g.resolveFutures(scc)

			// XXX this seems to be necessary, but shouldn't be.
			// removing it leads to nil pointer derefs; investigate
			// where we're not setting values correctly.
			for _, n := range vertices {
				if v, ok := n.Value.(ssa.Value); ok {
					i, ok := g.Range(v).(IntInterval)
					if !ok {
						continue
					}
					if !i.IsKnown() {
						g.SetRange(v, InfinityFor(v))
					}
				}
			}

			actives := g.actives(scc)
			for len(actives) > 0 {
				v := actives[len(actives)-1]
				actives = actives[:len(actives)-1]
				for _, use := range uses[v] {
					if g.narrow(use) {
						actives = append(actives, use.Y())
					}
				}
			}
		}
		// propagate scc
		for _, edge := range g.sccEdges[scc] {
			if edge.control {
				continue
			}
			if edge.From.SCC == edge.To.SCC {
				continue
			}
			if c, ok := edge.To.Value.(Constraint); ok {
				g.SetRange(c.Y(), c.Eval(g))
			}
			if c, ok := edge.To.Value.(Future); ok {
				if !c.IsKnown() {
					c.MarkUnresolved()
				}
			}
		}
	}

	for v, r := range g.ranges {
		i, ok := r.(IntInterval)
		if !ok {
			continue
		}
		if (v.Type().Underlying().(*types.Basic).Info() & types.IsUnsigned) == 0 {
			if i.Upper != PInfinity {
				s := &types.StdSizes{
					// XXX is it okay to assume the largest word size, or do we
					// need to be platform specific?
					WordSize: 8,
					MaxAlign: 1,
				}
				bits := (s.Sizeof(v.Type()) * 8) - 1
				n := big.NewInt(1)
				n = n.Lsh(n, uint(bits))
				upper, lower := &big.Int{}, &big.Int{}
				upper.Sub(n, big.NewInt(1))
				lower.Neg(n)

				if i.Upper.Cmp(NewBigZ(upper)) == 1 {
					i = NewIntInterval(NInfinity, PInfinity)
				} else if i.Lower.Cmp(NewBigZ(lower)) == -1 {
					i = NewIntInterval(NInfinity, PInfinity)
				}
			}
		}

		g.ranges[v] = i
	}

	return g.ranges
}

func VertexString(v *Vertex) string {
	switch v := v.Value.(type) {
	case Constraint:
		return v.String()
	case ssa.Value:
		return v.Name()
	case nil:
		return "BUG: nil vertex value"
	default:
		panic(fmt.Sprintf("unexpected type %T", v))
	}
}

type Vertex struct {
	Value   interface{} // one of Constraint or ssa.Value
	SCC     int
	index   int
	lowlink int
	stack   bool

	Succs []Edge
}

type Ranges map[ssa.Value]Range

func (r Ranges) Get(x ssa.Value) Range {
	if x == nil {
		return nil
	}
	i, ok := r[x]
	if !ok {
		switch x := x.Type().Underlying().(type) {
		case *types.Basic:
			switch x.Kind() {
			case types.String, types.UntypedString:
				return StringInterval{}
			default:
				return IntInterval{}
			}
		case *types.Chan:
			return ChannelInterval{}
		case *types.Slice:
			return SliceInterval{}
		}
	}
	return i
}

type Graph struct {
	Vertices map[interface{}]*Vertex
	Edges    []Edge
	SCCs     [][]*Vertex
	ranges   Ranges

	// map SCCs to futures
	futures [][]Future
	// map SCCs to edges
	sccEdges [][]Edge
}

func (g Graph) Graphviz() string {
	var lines []string
	lines = append(lines, "digraph{")
	ids := map[interface{}]int{}
	i := 1
	for _, v := range g.Vertices {
		ids[v] = i
		shape := "box"
		if _, ok := v.Value.(ssa.Value); ok {
			shape = "oval"
		}
		lines = append(lines, fmt.Sprintf(`n%d [shape="%s", label=%q, colorscheme=spectral11, style="filled", fillcolor="%d"]`,
			i, shape, VertexString(v), (v.SCC%11)+1))
		i++
	}
	for _, e := range g.Edges {
		style := "solid"
		if e.control {
			style = "dashed"
		}
		lines = append(lines, fmt.Sprintf(`n%d -> n%d [style="%s"]`, ids[e.From], ids[e.To], style))
	}
	lines = append(lines, "}")
	return strings.Join(lines, "\n")
}

func (g *Graph) SetRange(x ssa.Value, r Range) {
	g.ranges[x] = r
}

func (g *Graph) Range(x ssa.Value) Range {
	return g.ranges.Get(x)
}

func (g *Graph) widen(c Constraint, consts []Z) bool {
	setRange := func(i Range) {
		g.SetRange(c.Y(), i)
	}
	widenIntInterval := func(oi, ni IntInterval) (IntInterval, bool) {
		if !ni.IsKnown() {
			return oi, false
		}
		nlc := NInfinity
		nuc := PInfinity

		// Don't get stuck widening for an absurd amount of time due
		// to an excess number of constants, as may be present in
		// table-based scanners.
		if len(consts) < 1000 {
			for _, co := range consts {
				if co.Cmp(ni.Lower) <= 0 {
					nlc = co
					break
				}
			}
			for _, co := range consts {
				if co.Cmp(ni.Upper) >= 0 {
					nuc = co
					break
				}
			}
		}

		if !oi.IsKnown() {
			return ni, true
		}
		if ni.Lower.Cmp(oi.Lower) == -1 && ni.Upper.Cmp(oi.Upper) == 1 {
			return NewIntInterval(nlc, nuc), true
		}
		if ni.Lower.Cmp(oi.Lower) == -1 {
			return NewIntInterval(nlc, oi.Upper), true
		}
		if ni.Upper.Cmp(oi.Upper) == 1 {
			return NewIntInterval(oi.Lower, nuc), true
		}
		return oi, false
	}
	switch oi := g.Range(c.Y()).(type) {
	case IntInterval:
		ni := c.Eval(g).(IntInterval)
		si, changed := widenIntInterval(oi, ni)
		if changed {
			setRange(si)
			return true
		}
		return false
	case StringInterval:
		ni := c.Eval(g).(StringInterval)
		si, changed := widenIntInterval(oi.Length, ni.Length)
		if changed {
			setRange(StringInterval{si})
			return true
		}
		return false
	case SliceInterval:
		ni := c.Eval(g).(SliceInterval)
		si, changed := widenIntInterval(oi.Length, ni.Length)
		if changed {
			setRange(SliceInterval{si})
			return true
		}
		return false
	default:
		return false
	}
}

func (g *Graph) narrow(c Constraint) bool {
	narrowIntInterval := func(oi, ni IntInterval) (IntInterval, bool) {
		oLower := oi.Lower
		oUpper := oi.Upper
		nLower := ni.Lower
		nUpper := ni.Upper

		if oLower == NInfinity && nLower != NInfinity {
			return NewIntInterval(nLower, oUpper), true
		}
		if oUpper == PInfinity && nUpper != PInfinity {
			return NewIntInterval(oLower, nUpper), true
		}
		if oLower.Cmp(nLower) == 1 {
			return NewIntInterval(nLower, oUpper), true
		}
		if oUpper.Cmp(nUpper) == -1 {
			return NewIntInterval(oLower, nUpper), true
		}
		return oi, false
	}
	switch oi := g.Range(c.Y()).(type) {
	case IntInterval:
		ni := c.Eval(g).(IntInterval)
		si, changed := narrowIntInterval(oi, ni)
		if changed {
			g.SetRange(c.Y(), si)
			return true
		}
		return false
	case StringInterval:
		ni := c.Eval(g).(StringInterval)
		si, changed := narrowIntInterval(oi.Length, ni.Length)
		if changed {
			g.SetRange(c.Y(), StringInterval{si})
			return true
		}
		return false
	case SliceInterval:
		ni := c.Eval(g).(SliceInterval)
		si, changed := narrowIntInterval(oi.Length, ni.Length)
		if changed {
			g.SetRange(c.Y(), SliceInterval{si})
			return true
		}
		return false
	default:
		return false
	}
}

func (g *Graph) resolveFutures(scc int) {
	for _, c := range g.futures[scc] {
		c.Resolve()
	}
}

func (g *Graph) entries(scc int) []ssa.Value {
	var entries []ssa.Value
	for _, n := range g.Vertices {
		if n.SCC != scc {
			continue
		}
		if v, ok := n.Value.(ssa.Value); ok {
			// XXX avoid quadratic runtime
			//
			// XXX I cannot think of any code where the future and its
			// variables aren't in the same SCC, in which case this
			// code isn't very useful (the variables won't be resolved
			// yet). Before we have a cross-SCC example, however, we
			// can't really verify that this code is working
			// correctly, or indeed doing anything useful.
			for _, on := range g.Vertices {
				if c, ok := on.Value.(Future); ok {
					if c.Y() == v {
						if !c.IsResolved() {
							g.SetRange(c.Y(), c.Eval(g))
							c.MarkResolved()
						}
						break
					}
				}
			}
			if g.Range(v).IsKnown() {
				entries = append(entries, v)
			}
		}
	}
	return entries
}

func (g *Graph) uses(scc int) map[ssa.Value][]Constraint {
	m := map[ssa.Value][]Constraint{}
	for _, e := range g.sccEdges[scc] {
		if e.control {
			continue
		}
		if v, ok := e.From.Value.(ssa.Value); ok {
			c := e.To.Value.(Constraint)
			sink := c.Y()
			if g.Vertices[sink].SCC == scc {
				m[v] = append(m[v], c)
			}
		}
	}
	return m
}

func (g *Graph) actives(scc int) []ssa.Value {
	var actives []ssa.Value
	for _, n := range g.Vertices {
		if n.SCC != scc {
			continue
		}
		if v, ok := n.Value.(ssa.Value); ok {
			if _, ok := v.(*ssa.Const); !ok {
				actives = append(actives, v)
			}
		}
	}
	return actives
}

func (g *Graph) AddEdge(from, to interface{}, ctrl bool) {
	vf, ok := g.Vertices[from]
	if !ok {
		vf = &Vertex{Value: from}
		g.Vertices[from] = vf
	}
	vt, ok := g.Vertices[to]
	if !ok {
		vt = &Vertex{Value: to}
		g.Vertices[to] = vt
	}
	e := Edge{From: vf, To: vt, control: ctrl}
	g.Edges = append(g.Edges, e)
	vf.Succs = append(vf.Succs, e)
}

type Edge struct {
	From, To *Vertex
	control  bool
}

func (e Edge) String() string {
	return fmt.Sprintf("%s -> %s", VertexString(e.From), VertexString(e.To))
}

func (g *Graph) FindSCCs() {
	// use Tarjan to find the SCCs

	index := 1
	var s []*Vertex

	scc := 0
	var strongconnect func(v *Vertex)
	strongconnect = func(v *Vertex) {
		// set the depth index for v to the smallest unused index
		v.index = index
		v.lowlink = index
		index++
		s = append(s, v)
		v.stack = true

		for _, e := range v.Succs {
			w := e.To
			if w.index == 0 {
				// successor w has not yet been visited; recurse on it
				strongconnect(w)
				if w.lowlink < v.lowlink {
					v.lowlink = w.lowlink
				}
			} else if w.stack {
				// successor w is in stack s and hence in the current scc
				if w.index < v.lowlink {
					v.lowlink = w.index
				}
			}
		}

		if v.lowlink == v.index {
			for {
				w := s[len(s)-1]
				s = s[:len(s)-1]
				w.stack = false
				w.SCC = scc
				if w == v {
					break
				}
			}
			scc++
		}
	}
	for _, v := range g.Vertices {
		if v.index == 0 {
			strongconnect(v)
		}
	}

	g.SCCs = make([][]*Vertex, scc)
	for _, n := range g.Vertices {
		n.SCC = scc - n.SCC - 1
		g.SCCs[n.SCC] = append(g.SCCs[n.SCC], n)
	}
}

func invertToken(tok token.Token) token.Token {
	switch tok {
	case token.LSS:
		return token.GEQ
	case token.GTR:
		return token.LEQ
	case token.EQL:
		return token.NEQ
	case token.NEQ:
		return token.EQL
	case token.GEQ:
		return token.LSS
	case token.LEQ:
		return token.GTR
	default:
		panic(fmt.Sprintf("unsupported token %s", tok))
	}
}

func flipToken(tok token.Token) token.Token {
	switch tok {
	case token.LSS:
		return token.GTR
	case token.GTR:
		return token.LSS
	case token.EQL:
		return token.EQL
	case token.NEQ:
		return token.NEQ
	case token.GEQ:
		return token.LEQ
	case token.LEQ:
		return token.GEQ
	default:
		panic(fmt.Sprintf("unsupported token %s", tok))
	}
}

type CopyConstraint struct {
	aConstraint
	X ssa.Value
}

func (c *CopyConstraint) String() string {
	return fmt.Sprintf("%s = copy(%s)", c.Y().Name(), c.X.Name())
}

func (c *CopyConstraint) Eval(g *Graph) Range {
	return g.Range(c.X)
}

func (c *CopyConstraint) Operands() []ssa.Value {
	return []ssa.Value{c.X}
}

func NewCopyConstraint(x, y ssa.Value) Constraint {
	return &CopyConstraint{
		aConstraint: aConstraint{
			y: y,
		},
		X: x,
	}
}
