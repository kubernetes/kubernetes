//go:build windows

package computestorage

import (
	"context"
	"encoding/json"

	hcsschema "github.com/Microsoft/hcnshim/internal/hcs/schema2"
	"github.com/Microsoft/hcnshim/internal/oc"
	"github.com/pkg/errors"
	"go.opencensus.io/trace"
)

// DetachLayerStorageFilter detaches the layer storage filter on a writable container layer.
//
// `layerPath` is a path to a directory containing the layer to export.
func DetachLayerStorageFilter(ctx context.Context, layerPath string) (err error) {
	title := "hcnshim::DetachLayerStorageFilter"
	ctx, span := oc.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("layerPath", layerPath))

	err = hcsDetachLayerStorageFilter(layerPath)
	if err != nil {
		return errors.Wrap(err, "failed to detach layer storage filter")
	}
	return nil
}

// DetachOverlayFilter detaches the filter on a writable container layer.
//
// `volumePath` is a path to writable container volume.
func DetachOverlayFilter(ctx context.Context, volumePath string, filterType hcsschema.FileSystemFilterType) (err error) {
	title := "hcnshim::DetachOverlayFilter"
	ctx, span := oc.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("volumePath", volumePath))

	layerData := LayerData{}
	layerData.FilterType = filterType
	bytes, err := json.Marshal(layerData)
	if err != nil {
		return err
	}

	err = hcsDetachOverlayFilter(volumePath, string(bytes))
	if err != nil {
		return errors.Wrap(err, "failed to detach overlay filter")
	}
	return nil
}
