package netlink

import (
	"fmt"
	"net"
)

// Rule represents a netlink rule.
type Rule struct {
	Priority          int
	Family            int
	Table             int
	Mark              uint32
	Mask              *uint32
	Tos               uint
	TunID             uint
	Goto              int
	Src               *net.IPNet
	Dst               *net.IPNet
	Flow              int
	IifName           string
	OifName           string
	SuppressIfgroup   int
	SuppressPrefixlen int
	Invert            bool
	Dport             *RulePortRange
	Sport             *RulePortRange
	IPProto           int
	UIDRange          *RuleUIDRange
	Protocol          uint8
	Type              uint8
}

func (r Rule) String() string {
	from := "all"
	if r.Src != nil && r.Src.String() != "<nil>" {
		from = r.Src.String()
	}

	to := "all"
	if r.Dst != nil && r.Dst.String() != "<nil>" {
		to = r.Dst.String()
	}

	return fmt.Sprintf("ip rule %d: from %s to %s table %d %s",
		r.Priority, from, to, r.Table, r.typeString())
}

// NewRule return empty rules.
func NewRule() *Rule {
	return &Rule{
		SuppressIfgroup:   -1,
		SuppressPrefixlen: -1,
		Priority:          -1,
		Mark:              0,
		Mask:              nil,
		Goto:              -1,
		Flow:              -1,
	}
}

// NewRulePortRange creates rule sport/dport range.
func NewRulePortRange(start, end uint16) *RulePortRange {
	return &RulePortRange{Start: start, End: end}
}

// RulePortRange represents rule sport/dport range.
type RulePortRange struct {
	Start uint16
	End   uint16
}

// NewRuleUIDRange creates rule uid range.
func NewRuleUIDRange(start, end uint32) *RuleUIDRange {
	return &RuleUIDRange{Start: start, End: end}
}

// RuleUIDRange represents rule uid range.
type RuleUIDRange struct {
	Start uint32
	End   uint32
}
