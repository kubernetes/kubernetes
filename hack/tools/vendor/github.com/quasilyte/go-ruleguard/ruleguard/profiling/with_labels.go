//go:build pproflabels
// +build pproflabels

package profiling

import (
	"context"
	"runtime/pprof"
)

const LabelsEnabled = true

func EnterWithLabels(origContext context.Context, name string) {
	labels := pprof.Labels("rules", name)
	ctx := pprof.WithLabels(origContext, labels)
	pprof.SetGoroutineLabels(ctx)
}

func Leave(origContext context.Context) {
	pprof.SetGoroutineLabels(origContext)
}
