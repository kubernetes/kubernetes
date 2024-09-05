//go:build windows

package wclayer

import (
	"context"
	"strings"
	"sync"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
)

var prepareLayerLock sync.Mutex

// PrepareLayer finds a mounted read-write layer matching path and enables the
// the filesystem filter for use on that layer.  This requires the paths to all
// parent layers, and is necessary in order to view or interact with the layer
// as an actual filesystem (reading and writing files, creating directories, etc).
// Disabling the filter must be done via UnprepareLayer.
func PrepareLayer(ctx context.Context, path string, parentLayerPaths []string) (err error) {
	title := "hcsshim::PrepareLayer"
	ctx, span := oc.StartSpan(ctx, title)
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

	// This lock is a temporary workaround for a Windows bug. Only allowing one
	// call to prepareLayer at a time vastly reduces the chance of a timeout.
	prepareLayerLock.Lock()
	defer prepareLayerLock.Unlock()
	err = prepareLayer(&stdDriverInfo, path, layers)
	if err != nil {
		return hcserror.New(err, title, "")
	}
	return nil
}
