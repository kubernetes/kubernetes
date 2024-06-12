//go:build windows

package wclayer

import (
	"context"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
)

// ActivateLayer will find the layer with the given id and mount it's filesystem.
// For a read/write layer, the mounted filesystem will appear as a volume on the
// host, while a read-only layer is generally expected to be a no-op.
// An activated layer must later be deactivated via DeactivateLayer.
func ActivateLayer(ctx context.Context, path string) (err error) {
	title := "hcsshim::ActivateLayer"
	ctx, span := oc.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("path", path))

	err = activateLayer(&stdDriverInfo, path)
	if err != nil {
		return hcserror.New(err, title, "")
	}
	return nil
}
