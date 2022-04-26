//go:build !pproflabels
// +build !pproflabels

package profiling

import (
	"context"
)

const LabelsEnabled = false

func EnterWithLabels(origContext context.Context, name string) {
}

func Leave(origContext context.Context) {
}
