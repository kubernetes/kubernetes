package computestorage

import (
	"context"

	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/pkg/errors"
	"go.opencensus.io/trace"
)

// DetachLayerStorageFilter detaches the layer storage filter on a writable container layer.
//
// `layerPath` is a path to a directory containing the layer to export.
func DetachLayerStorageFilter(ctx context.Context, layerPath string) (err error) {
	title := "hcsshim.DetachLayerStorageFilter"
	ctx, span := trace.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("layerPath", layerPath))

	err = hcsDetachLayerStorageFilter(layerPath)
	if err != nil {
		return errors.Wrap(err, "failed to detach layer storage filter")
	}
	return nil
}
