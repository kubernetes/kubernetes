//go:build windows

package wclayer

import (
	"context"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
)

// UnprepareLayer disables the filesystem filter for the read-write layer with
// the given id.
func UnprepareLayer(ctx context.Context, path string) (err error) {
	title := "hcsshim::UnprepareLayer"
	ctx, span := oc.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("path", path))

	err = unprepareLayer(&stdDriverInfo, path)
	if err != nil {
		return hcserror.New(err, title, "")
	}
	return nil
}
