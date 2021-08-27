package wclayer

import (
	"context"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
)

// LayerExists will return true if a layer with the given id exists and is known
// to the system.
func LayerExists(ctx context.Context, path string) (_ bool, err error) {
	title := "hcsshim::LayerExists"
	ctx, span := trace.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("path", path))

	// Call the procedure itself.
	var exists uint32
	err = layerExists(&stdDriverInfo, path, &exists)
	if err != nil {
		return false, hcserror.New(err, title+" - failed", "")
	}
	span.AddAttributes(trace.BoolAttribute("layer-exists", exists != 0))
	return exists != 0, nil
}
