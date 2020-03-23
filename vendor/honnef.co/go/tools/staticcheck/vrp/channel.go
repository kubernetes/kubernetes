package vrp

import (
	"fmt"

	"honnef.co/go/tools/ssa"
)

type ChannelInterval struct {
	Size IntInterval
}

func (c ChannelInterval) Union(other Range) Range {
	i, ok := other.(ChannelInterval)
	if !ok {
		i = ChannelInterval{EmptyIntInterval}
	}
	if c.Size.Empty() || !c.Size.IsKnown() {
		return i
	}
	if i.Size.Empty() || !i.Size.IsKnown() {
		return c
	}
	return ChannelInterval{
		Size: c.Size.Union(i.Size).(IntInterval),
	}
}

func (c ChannelInterval) String() string {
	return c.Size.String()
}

func (c ChannelInterval) IsKnown() bool {
	return c.Size.IsKnown()
}

type MakeChannelConstraint struct {
	aConstraint
	Buffer ssa.Value
}
type ChannelChangeTypeConstraint struct {
	aConstraint
	X ssa.Value
}

func NewMakeChannelConstraint(buffer, y ssa.Value) Constraint {
	return &MakeChannelConstraint{NewConstraint(y), buffer}
}
func NewChannelChangeTypeConstraint(x, y ssa.Value) Constraint {
	return &ChannelChangeTypeConstraint{NewConstraint(y), x}
}

func (c *MakeChannelConstraint) Operands() []ssa.Value       { return []ssa.Value{c.Buffer} }
func (c *ChannelChangeTypeConstraint) Operands() []ssa.Value { return []ssa.Value{c.X} }

func (c *MakeChannelConstraint) String() string {
	return fmt.Sprintf("%s = make(chan, %s)", c.Y().Name(), c.Buffer.Name())
}
func (c *ChannelChangeTypeConstraint) String() string {
	return fmt.Sprintf("%s = changetype(%s)", c.Y().Name(), c.X.Name())
}

func (c *MakeChannelConstraint) Eval(g *Graph) Range {
	i, ok := g.Range(c.Buffer).(IntInterval)
	if !ok {
		return ChannelInterval{NewIntInterval(NewZ(0), PInfinity)}
	}
	if i.Lower.Sign() == -1 {
		i.Lower = NewZ(0)
	}
	return ChannelInterval{i}
}
func (c *ChannelChangeTypeConstraint) Eval(g *Graph) Range { return g.Range(c.X) }
