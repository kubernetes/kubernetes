package types

import (
	"fmt"
	"strings"

	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

type formattable interface {
	format(*strings.Builder)
}

// Format formats the value as a string. The result is only intended for human consumption and ignores errors.
// Do not depend on the output being stable. It may change at any time.
func Format(val ref.Val) string {
	var sb strings.Builder
	formatTo(&sb, val)
	return sb.String()
}

func formatTo(sb *strings.Builder, val ref.Val) {
	if fmtable, ok := val.(formattable); ok {
		fmtable.format(sb)
		return
	}
	// All of the builtins implement formattable. Try to deal with traits.
	if l, ok := val.(traits.Lister); ok {
		formatList(l, sb)
		return
	}
	if m, ok := val.(traits.Mapper); ok {
		formatMap(m, sb)
		return
	}
	// This could be an error, unknown, opaque or object.
	// Unfortunately we have no consistent way of inspecting
	// opaque and object. So we just fallback to fmt.Stringer
	// and hope it is relavent.
	fmt.Fprintf(sb, "%s", val)
}
