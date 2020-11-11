// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

// This file implements the String() methods for all Value and
// Instruction types.

import (
	"bytes"
	"fmt"
	"go/types"
	"io"
	"reflect"
	"sort"

	"golang.org/x/tools/go/types/typeutil"
)

// relName returns the name of v relative to i.
// In most cases, this is identical to v.Name(), but references to
// Functions (including methods) and Globals use RelString and
// all types are displayed with relType, so that only cross-package
// references are package-qualified.
//
func relName(v Value, i Instruction) string {
	if v == nil {
		return "<nil>"
	}
	var from *types.Package
	if i != nil {
		from = i.Parent().pkg()
	}
	switch v := v.(type) {
	case Member: // *Function or *Global
		return v.RelString(from)
	}
	return v.Name()
}

func relType(t types.Type, from *types.Package) string {
	return types.TypeString(t, types.RelativeTo(from))
}

func relString(m Member, from *types.Package) string {
	// NB: not all globals have an Object (e.g. init$guard),
	// so use Package().Object not Object.Package().
	if pkg := m.Package().Pkg; pkg != nil && pkg != from {
		return fmt.Sprintf("%s.%s", pkg.Path(), m.Name())
	}
	return m.Name()
}

// Value.String()
//
// This method is provided only for debugging.
// It never appears in disassembly, which uses Value.Name().

func (v *Parameter) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("Parameter <%s> {%s}", relType(v.Type(), from), v.name)
}

func (v *FreeVar) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("FreeVar <%s> %s", relType(v.Type(), from), v.Name())
}

func (v *Builtin) String() string {
	return fmt.Sprintf("Builtin %s", v.Name())
}

// Instruction.String()

func (v *Alloc) String() string {
	from := v.Parent().pkg()
	storage := "Stack"
	if v.Heap {
		storage = "Heap"
	}
	return fmt.Sprintf("%sAlloc <%s>", storage, relType(v.Type(), from))
}

func (v *Sigma) String() string {
	from := v.Parent().pkg()
	s := fmt.Sprintf("Sigma <%s> [b%d] %s", relType(v.Type(), from), v.From.Index, v.X.Name())
	return s
}

func (v *Phi) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "Phi <%s>", v.Type())
	for i, edge := range v.Edges {
		b.WriteString(" ")
		// Be robust against malformed CFG.
		if v.block == nil {
			b.WriteString("??")
			continue
		}
		block := -1
		if i < len(v.block.Preds) {
			block = v.block.Preds[i].Index
		}
		fmt.Fprintf(&b, "%d:", block)
		edgeVal := "<nil>" // be robust
		if edge != nil {
			edgeVal = relName(edge, v)
		}
		b.WriteString(edgeVal)
	}
	return b.String()
}

func printCall(v *CallCommon, prefix string, instr Instruction) string {
	var b bytes.Buffer
	if !v.IsInvoke() {
		if value, ok := instr.(Value); ok {
			fmt.Fprintf(&b, "%s <%s> %s", prefix, relType(value.Type(), instr.Parent().pkg()), relName(v.Value, instr))
		} else {
			fmt.Fprintf(&b, "%s %s", prefix, relName(v.Value, instr))
		}
	} else {
		if value, ok := instr.(Value); ok {
			fmt.Fprintf(&b, "%sInvoke <%s> %s.%s", prefix, relType(value.Type(), instr.Parent().pkg()), relName(v.Value, instr), v.Method.Name())
		} else {
			fmt.Fprintf(&b, "%sInvoke %s.%s", prefix, relName(v.Value, instr), v.Method.Name())
		}
	}
	for _, arg := range v.Args {
		b.WriteString(" ")
		b.WriteString(relName(arg, instr))
	}
	return b.String()
}

func (c *CallCommon) String() string {
	return printCall(c, "", nil)
}

func (v *Call) String() string {
	return printCall(&v.Call, "Call", v)
}

func (v *BinOp) String() string {
	return fmt.Sprintf("BinOp <%s> {%s} %s %s", relType(v.Type(), v.Parent().pkg()), v.Op.String(), relName(v.X, v), relName(v.Y, v))
}

func (v *UnOp) String() string {
	return fmt.Sprintf("UnOp <%s> {%s} %s", relType(v.Type(), v.Parent().pkg()), v.Op.String(), relName(v.X, v))
}

func (v *Load) String() string {
	return fmt.Sprintf("Load <%s> %s", relType(v.Type(), v.Parent().pkg()), relName(v.X, v))
}

func printConv(prefix string, v, x Value) string {
	from := v.Parent().pkg()
	return fmt.Sprintf("%s <%s> %s",
		prefix,
		relType(v.Type(), from),
		relName(x, v.(Instruction)))
}

func (v *ChangeType) String() string      { return printConv("ChangeType", v, v.X) }
func (v *Convert) String() string         { return printConv("Convert", v, v.X) }
func (v *ChangeInterface) String() string { return printConv("ChangeInterface", v, v.X) }
func (v *MakeInterface) String() string   { return printConv("MakeInterface", v, v.X) }

func (v *MakeClosure) String() string {
	from := v.Parent().pkg()
	var b bytes.Buffer
	fmt.Fprintf(&b, "MakeClosure <%s> %s", relType(v.Type(), from), relName(v.Fn, v))
	if v.Bindings != nil {
		for _, c := range v.Bindings {
			b.WriteString(" ")
			b.WriteString(relName(c, v))
		}
	}
	return b.String()
}

func (v *MakeSlice) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("MakeSlice <%s> %s %s",
		relType(v.Type(), from),
		relName(v.Len, v),
		relName(v.Cap, v))
}

func (v *Slice) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("Slice <%s> %s %s %s %s",
		relType(v.Type(), from), relName(v.X, v), relName(v.Low, v), relName(v.High, v), relName(v.Max, v))
}

func (v *MakeMap) String() string {
	res := ""
	if v.Reserve != nil {
		res = relName(v.Reserve, v)
	}
	from := v.Parent().pkg()
	return fmt.Sprintf("MakeMap <%s> %s", relType(v.Type(), from), res)
}

func (v *MakeChan) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("MakeChan <%s> %s", relType(v.Type(), from), relName(v.Size, v))
}

func (v *FieldAddr) String() string {
	from := v.Parent().pkg()
	st := deref(v.X.Type()).Underlying().(*types.Struct)
	// Be robust against a bad index.
	name := "?"
	if 0 <= v.Field && v.Field < st.NumFields() {
		name = st.Field(v.Field).Name()
	}
	return fmt.Sprintf("FieldAddr <%s> [%d] (%s) %s", relType(v.Type(), from), v.Field, name, relName(v.X, v))
}

func (v *Field) String() string {
	st := v.X.Type().Underlying().(*types.Struct)
	// Be robust against a bad index.
	name := "?"
	if 0 <= v.Field && v.Field < st.NumFields() {
		name = st.Field(v.Field).Name()
	}
	from := v.Parent().pkg()
	return fmt.Sprintf("Field <%s> [%d] (%s) %s", relType(v.Type(), from), v.Field, name, relName(v.X, v))
}

func (v *IndexAddr) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("IndexAddr <%s> %s %s", relType(v.Type(), from), relName(v.X, v), relName(v.Index, v))
}

func (v *Index) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("Index <%s> %s %s", relType(v.Type(), from), relName(v.X, v), relName(v.Index, v))
}

func (v *MapLookup) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("MapLookup <%s> %s %s", relType(v.Type(), from), relName(v.X, v), relName(v.Index, v))
}

func (v *StringLookup) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("StringLookup <%s> %s %s", relType(v.Type(), from), relName(v.X, v), relName(v.Index, v))
}

func (v *Range) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("Range <%s> %s", relType(v.Type(), from), relName(v.X, v))
}

func (v *Next) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("Next <%s> %s", relType(v.Type(), from), relName(v.Iter, v))
}

func (v *TypeAssert) String() string {
	from := v.Parent().pkg()
	return fmt.Sprintf("TypeAssert <%s> %s", relType(v.Type(), from), relName(v.X, v))
}

func (v *Extract) String() string {
	from := v.Parent().pkg()
	name := v.Tuple.Type().(*types.Tuple).At(v.Index).Name()
	return fmt.Sprintf("Extract <%s> [%d] (%s) %s", relType(v.Type(), from), v.Index, name, relName(v.Tuple, v))
}

func (s *Jump) String() string {
	// Be robust against malformed CFG.
	block := -1
	if s.block != nil && len(s.block.Succs) == 1 {
		block = s.block.Succs[0].Index
	}
	str := fmt.Sprintf("Jump → b%d", block)
	if s.Comment != "" {
		str = fmt.Sprintf("%s # %s", str, s.Comment)
	}
	return str
}

func (s *Unreachable) String() string {
	// Be robust against malformed CFG.
	block := -1
	if s.block != nil && len(s.block.Succs) == 1 {
		block = s.block.Succs[0].Index
	}
	return fmt.Sprintf("Unreachable → b%d", block)
}

func (s *If) String() string {
	// Be robust against malformed CFG.
	tblock, fblock := -1, -1
	if s.block != nil && len(s.block.Succs) == 2 {
		tblock = s.block.Succs[0].Index
		fblock = s.block.Succs[1].Index
	}
	return fmt.Sprintf("If %s → b%d b%d", relName(s.Cond, s), tblock, fblock)
}

func (s *ConstantSwitch) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "ConstantSwitch %s", relName(s.Tag, s))
	for _, cond := range s.Conds {
		fmt.Fprintf(&b, " %s", relName(cond, s))
	}
	fmt.Fprint(&b, " →")
	for _, succ := range s.block.Succs {
		fmt.Fprintf(&b, " b%d", succ.Index)
	}
	return b.String()
}

func (s *TypeSwitch) String() string {
	from := s.Parent().pkg()
	var b bytes.Buffer
	fmt.Fprintf(&b, "TypeSwitch <%s> %s", relType(s.typ, from), relName(s.Tag, s))
	for _, cond := range s.Conds {
		fmt.Fprintf(&b, " %q", relType(cond, s.block.parent.pkg()))
	}
	return b.String()
}

func (s *Go) String() string {
	return printCall(&s.Call, "Go", s)
}

func (s *Panic) String() string {
	// Be robust against malformed CFG.
	block := -1
	if s.block != nil && len(s.block.Succs) == 1 {
		block = s.block.Succs[0].Index
	}
	return fmt.Sprintf("Panic %s → b%d", relName(s.X, s), block)
}

func (s *Return) String() string {
	var b bytes.Buffer
	b.WriteString("Return")
	for _, r := range s.Results {
		b.WriteString(" ")
		b.WriteString(relName(r, s))
	}
	return b.String()
}

func (*RunDefers) String() string {
	return "RunDefers"
}

func (s *Send) String() string {
	return fmt.Sprintf("Send %s %s", relName(s.Chan, s), relName(s.X, s))
}

func (recv *Recv) String() string {
	from := recv.Parent().pkg()
	return fmt.Sprintf("Recv <%s> %s", relType(recv.Type(), from), relName(recv.Chan, recv))
}

func (s *Defer) String() string {
	return printCall(&s.Call, "Defer", s)
}

func (s *Select) String() string {
	var b bytes.Buffer
	for i, st := range s.States {
		if i > 0 {
			b.WriteString(", ")
		}
		if st.Dir == types.RecvOnly {
			b.WriteString("<-")
			b.WriteString(relName(st.Chan, s))
		} else {
			b.WriteString(relName(st.Chan, s))
			b.WriteString("<-")
			b.WriteString(relName(st.Send, s))
		}
	}
	non := ""
	if !s.Blocking {
		non = "Non"
	}
	from := s.Parent().pkg()
	return fmt.Sprintf("Select%sBlocking <%s> [%s]", non, relType(s.Type(), from), b.String())
}

func (s *Store) String() string {
	return fmt.Sprintf("Store {%s} %s %s",
		s.Val.Type(), relName(s.Addr, s), relName(s.Val, s))
}

func (s *BlankStore) String() string {
	return fmt.Sprintf("BlankStore %s", relName(s.Val, s))
}

func (s *MapUpdate) String() string {
	return fmt.Sprintf("MapUpdate %s %s %s", relName(s.Map, s), relName(s.Key, s), relName(s.Value, s))
}

func (s *DebugRef) String() string {
	p := s.Parent().Prog.Fset.Position(s.Pos())
	var descr interface{}
	if s.object != nil {
		descr = s.object // e.g. "var x int"
	} else {
		descr = reflect.TypeOf(s.Expr) // e.g. "*ast.CallExpr"
	}
	var addr string
	if s.IsAddr {
		addr = "address of "
	}
	return fmt.Sprintf("; %s%s @ %d:%d is %s", addr, descr, p.Line, p.Column, s.X.Name())
}

func (p *Package) String() string {
	return "package " + p.Pkg.Path()
}

var _ io.WriterTo = (*Package)(nil) // *Package implements io.Writer

func (p *Package) WriteTo(w io.Writer) (int64, error) {
	var buf bytes.Buffer
	WritePackage(&buf, p)
	n, err := w.Write(buf.Bytes())
	return int64(n), err
}

// WritePackage writes to buf a human-readable summary of p.
func WritePackage(buf *bytes.Buffer, p *Package) {
	fmt.Fprintf(buf, "%s:\n", p)

	var names []string
	maxname := 0
	for name := range p.Members {
		if l := len(name); l > maxname {
			maxname = l
		}
		names = append(names, name)
	}

	from := p.Pkg
	sort.Strings(names)
	for _, name := range names {
		switch mem := p.Members[name].(type) {
		case *NamedConst:
			fmt.Fprintf(buf, "  const %-*s %s = %s\n",
				maxname, name, mem.Name(), mem.Value.RelString(from))

		case *Function:
			fmt.Fprintf(buf, "  func  %-*s %s\n",
				maxname, name, relType(mem.Type(), from))

		case *Type:
			fmt.Fprintf(buf, "  type  %-*s %s\n",
				maxname, name, relType(mem.Type().Underlying(), from))
			for _, meth := range typeutil.IntuitiveMethodSet(mem.Type(), &p.Prog.MethodSets) {
				fmt.Fprintf(buf, "    %s\n", types.SelectionString(meth, types.RelativeTo(from)))
			}

		case *Global:
			fmt.Fprintf(buf, "  var   %-*s %s\n",
				maxname, name, relType(mem.Type().(*types.Pointer).Elem(), from))
		}
	}

	fmt.Fprintf(buf, "\n")
}
