package vrp

// TODO(dh): most of the constraints have implementations identical to
// that of strings. Consider reusing them.

import (
	"fmt"
	"go/types"

	"honnef.co/go/tools/ssa"
)

type SliceInterval struct {
	Length IntInterval
}

func (s SliceInterval) Union(other Range) Range {
	i, ok := other.(SliceInterval)
	if !ok {
		i = SliceInterval{EmptyIntInterval}
	}
	if s.Length.Empty() || !s.Length.IsKnown() {
		return i
	}
	if i.Length.Empty() || !i.Length.IsKnown() {
		return s
	}
	return SliceInterval{
		Length: s.Length.Union(i.Length).(IntInterval),
	}
}
func (s SliceInterval) String() string { return s.Length.String() }
func (s SliceInterval) IsKnown() bool  { return s.Length.IsKnown() }

type SliceAppendConstraint struct {
	aConstraint
	A ssa.Value
	B ssa.Value
}

type SliceSliceConstraint struct {
	aConstraint
	X     ssa.Value
	Lower ssa.Value
	Upper ssa.Value
}

type ArraySliceConstraint struct {
	aConstraint
	X     ssa.Value
	Lower ssa.Value
	Upper ssa.Value
}

type SliceIntersectionConstraint struct {
	aConstraint
	X ssa.Value
	I IntInterval
}

type SliceLengthConstraint struct {
	aConstraint
	X ssa.Value
}

type MakeSliceConstraint struct {
	aConstraint
	Size ssa.Value
}

type SliceIntervalConstraint struct {
	aConstraint
	I IntInterval
}

func NewSliceAppendConstraint(a, b, y ssa.Value) Constraint {
	return &SliceAppendConstraint{NewConstraint(y), a, b}
}
func NewSliceSliceConstraint(x, lower, upper, y ssa.Value) Constraint {
	return &SliceSliceConstraint{NewConstraint(y), x, lower, upper}
}
func NewArraySliceConstraint(x, lower, upper, y ssa.Value) Constraint {
	return &ArraySliceConstraint{NewConstraint(y), x, lower, upper}
}
func NewSliceIntersectionConstraint(x ssa.Value, i IntInterval, y ssa.Value) Constraint {
	return &SliceIntersectionConstraint{NewConstraint(y), x, i}
}
func NewSliceLengthConstraint(x, y ssa.Value) Constraint {
	return &SliceLengthConstraint{NewConstraint(y), x}
}
func NewMakeSliceConstraint(size, y ssa.Value) Constraint {
	return &MakeSliceConstraint{NewConstraint(y), size}
}
func NewSliceIntervalConstraint(i IntInterval, y ssa.Value) Constraint {
	return &SliceIntervalConstraint{NewConstraint(y), i}
}

func (c *SliceAppendConstraint) Operands() []ssa.Value { return []ssa.Value{c.A, c.B} }
func (c *SliceSliceConstraint) Operands() []ssa.Value {
	ops := []ssa.Value{c.X}
	if c.Lower != nil {
		ops = append(ops, c.Lower)
	}
	if c.Upper != nil {
		ops = append(ops, c.Upper)
	}
	return ops
}
func (c *ArraySliceConstraint) Operands() []ssa.Value {
	ops := []ssa.Value{c.X}
	if c.Lower != nil {
		ops = append(ops, c.Lower)
	}
	if c.Upper != nil {
		ops = append(ops, c.Upper)
	}
	return ops
}
func (c *SliceIntersectionConstraint) Operands() []ssa.Value { return []ssa.Value{c.X} }
func (c *SliceLengthConstraint) Operands() []ssa.Value       { return []ssa.Value{c.X} }
func (c *MakeSliceConstraint) Operands() []ssa.Value         { return []ssa.Value{c.Size} }
func (s *SliceIntervalConstraint) Operands() []ssa.Value     { return nil }

func (c *SliceAppendConstraint) String() string {
	return fmt.Sprintf("%s = append(%s, %s)", c.Y().Name(), c.A.Name(), c.B.Name())
}
func (c *SliceSliceConstraint) String() string {
	var lname, uname string
	if c.Lower != nil {
		lname = c.Lower.Name()
	}
	if c.Upper != nil {
		uname = c.Upper.Name()
	}
	return fmt.Sprintf("%s[%s:%s]", c.X.Name(), lname, uname)
}
func (c *ArraySliceConstraint) String() string {
	var lname, uname string
	if c.Lower != nil {
		lname = c.Lower.Name()
	}
	if c.Upper != nil {
		uname = c.Upper.Name()
	}
	return fmt.Sprintf("%s[%s:%s]", c.X.Name(), lname, uname)
}
func (c *SliceIntersectionConstraint) String() string {
	return fmt.Sprintf("%s = %s.%t ⊓ %s", c.Y().Name(), c.X.Name(), c.Y().(*ssa.Sigma).Branch, c.I)
}
func (c *SliceLengthConstraint) String() string {
	return fmt.Sprintf("%s = len(%s)", c.Y().Name(), c.X.Name())
}
func (c *MakeSliceConstraint) String() string {
	return fmt.Sprintf("%s = make(slice, %s)", c.Y().Name(), c.Size.Name())
}
func (c *SliceIntervalConstraint) String() string { return fmt.Sprintf("%s = %s", c.Y().Name(), c.I) }

func (c *SliceAppendConstraint) Eval(g *Graph) Range {
	l1 := g.Range(c.A).(SliceInterval).Length
	var l2 IntInterval
	switch r := g.Range(c.B).(type) {
	case SliceInterval:
		l2 = r.Length
	case StringInterval:
		l2 = r.Length
	default:
		return SliceInterval{}
	}
	if !l1.IsKnown() || !l2.IsKnown() {
		return SliceInterval{}
	}
	return SliceInterval{
		Length: l1.Add(l2),
	}
}
func (c *SliceSliceConstraint) Eval(g *Graph) Range {
	lr := NewIntInterval(NewZ(0), NewZ(0))
	if c.Lower != nil {
		lr = g.Range(c.Lower).(IntInterval)
	}
	ur := g.Range(c.X).(SliceInterval).Length
	if c.Upper != nil {
		ur = g.Range(c.Upper).(IntInterval)
	}
	if !lr.IsKnown() || !ur.IsKnown() {
		return SliceInterval{}
	}

	ls := []Z{
		ur.Lower.Sub(lr.Lower),
		ur.Upper.Sub(lr.Lower),
		ur.Lower.Sub(lr.Upper),
		ur.Upper.Sub(lr.Upper),
	}
	// TODO(dh): if we don't truncate lengths to 0 we might be able to
	// easily detect slices with high < low. we'd need to treat -∞
	// specially, though.
	for i, l := range ls {
		if l.Sign() == -1 {
			ls[i] = NewZ(0)
		}
	}

	return SliceInterval{
		Length: NewIntInterval(MinZ(ls...), MaxZ(ls...)),
	}
}
func (c *ArraySliceConstraint) Eval(g *Graph) Range {
	lr := NewIntInterval(NewZ(0), NewZ(0))
	if c.Lower != nil {
		lr = g.Range(c.Lower).(IntInterval)
	}
	var l int64
	switch typ := c.X.Type().(type) {
	case *types.Array:
		l = typ.Len()
	case *types.Pointer:
		l = typ.Elem().(*types.Array).Len()
	}
	ur := NewIntInterval(NewZ(l), NewZ(l))
	if c.Upper != nil {
		ur = g.Range(c.Upper).(IntInterval)
	}
	if !lr.IsKnown() || !ur.IsKnown() {
		return SliceInterval{}
	}

	ls := []Z{
		ur.Lower.Sub(lr.Lower),
		ur.Upper.Sub(lr.Lower),
		ur.Lower.Sub(lr.Upper),
		ur.Upper.Sub(lr.Upper),
	}
	// TODO(dh): if we don't truncate lengths to 0 we might be able to
	// easily detect slices with high < low. we'd need to treat -∞
	// specially, though.
	for i, l := range ls {
		if l.Sign() == -1 {
			ls[i] = NewZ(0)
		}
	}

	return SliceInterval{
		Length: NewIntInterval(MinZ(ls...), MaxZ(ls...)),
	}
}
func (c *SliceIntersectionConstraint) Eval(g *Graph) Range {
	xi := g.Range(c.X).(SliceInterval)
	if !xi.IsKnown() {
		return c.I
	}
	return SliceInterval{
		Length: xi.Length.Intersection(c.I),
	}
}
func (c *SliceLengthConstraint) Eval(g *Graph) Range {
	i := g.Range(c.X).(SliceInterval).Length
	if !i.IsKnown() {
		return NewIntInterval(NewZ(0), PInfinity)
	}
	return i
}
func (c *MakeSliceConstraint) Eval(g *Graph) Range {
	i, ok := g.Range(c.Size).(IntInterval)
	if !ok {
		return SliceInterval{NewIntInterval(NewZ(0), PInfinity)}
	}
	if i.Lower.Sign() == -1 {
		i.Lower = NewZ(0)
	}
	return SliceInterval{i}
}
func (c *SliceIntervalConstraint) Eval(*Graph) Range { return SliceInterval{c.I} }
