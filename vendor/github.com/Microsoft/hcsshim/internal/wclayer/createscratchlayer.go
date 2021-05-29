package wclayer

import (
	"context"
	"strings"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
)

// CreateScratchLayer creates and populates new read-write layer for use by a container.
// This requires both the id of the direct parent layer, as well as the full list
// of paths to all parent layers up to the base (and including the direct parent
// whose id was provided).
func CreateScratchLayer(ctx context.Context, path string, parentLayerPaths []string) (err error) {
	title := "hcsshim::CreateScratchLayer"
	ctx, span := trace.StartSpan(ctx, title)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(
		trace.StringAttribute("path", path),
		trace.StringAttribute("parentLayerPaths", strings.Join(parentLayerPaths, ", ")))

	// Generate layer descriptors
	layers, err := layerPathsToDescriptors(ctx, parentLayerPaths)
	if err != nil {
		return err
	}

	err = createSandboxLayer(&stdDriverInfo, path, 0, layers)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
