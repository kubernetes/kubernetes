package netlink

import (
	"strings"
)

// Protinfo represents bridge flags from netlink.
type Protinfo struct {
	Hairpin   bool
	Guard     bool
	FastLeave bool
	RootBlock bool
	Learning  bool
	Flood     bool
}

// String returns a list of enabled flags
func (prot *Protinfo) String() string {
	var boolStrings []string
	if prot.Hairpin {
		boolStrings = append(boolStrings, "Hairpin")
	}
	if prot.Guard {
		boolStrings = append(boolStrings, "Guard")
	}
	if prot.FastLeave {
		boolStrings = append(boolStrings, "FastLeave")
	}
	if prot.RootBlock {
		boolStrings = append(boolStrings, "RootBlock")
	}
	if prot.Learning {
		boolStrings = append(boolStrings, "Learning")
	}
	if prot.Flood {
		boolStrings = append(boolStrings, "Flood")
	}
	return strings.Join(boolStrings, " ")
}

func boolToByte(x bool) []byte {
	if x {
		return []byte{1}
	}
	return []byte{0}
}

func byteToBool(x byte) bool {
	if uint8(x) != 0 {
		return true
	}
	return false
}
