package wclayer

import (
	"context"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
)

// DestroyLayer will remove the on-disk files representing the layer with the given
// path, including that layer's containing folder, if any.
func DestroyLayer(ctx context.Context, path string) (err error) {
	title := "hcsshim::DestroyLayer"
	ctx, span := trace.StartSpan(ctx, title)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("path", path))

	err = destroyLayer(&stdDriverInfo, path)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
