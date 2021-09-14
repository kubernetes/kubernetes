package computestorage

import (
	"context"

	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/pkg/errors"
	"go.opencensus.io/trace"
)

// DestroyLayer deletes a container layer.
//
// `layerPath` is a path to a directory containing the layer to export.
func DestroyLayer(ctx context.Context, layerPath string) (err error) {
	title := "hcsshim.DestroyLayer"
	ctx, span := trace.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("layerPath", layerPath))

	err = hcsDestroyLayer(layerPath)
	if err != nil {
		return errors.Wrap(err, "failed to destroy layer")
	}
	return nil
}
