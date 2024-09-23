//go:build windows

package computestorage

import (
	"context"
	"encoding/json"

	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/pkg/errors"
	"go.opencensus.io/trace"
)

// InitializeWritableLayer initializes a writable layer for a container.
//
// `layerPath` is a path to a directory the layer is mounted. If the
// path does not end in a `\` the platform will append it automatically.
//
// `layerData` is the parent read-only layer data.
func InitializeWritableLayer(ctx context.Context, layerPath string, layerData LayerData) (err error) {
	title := "hcsshim::InitializeWritableLayer"
	ctx, span := oc.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(
		trace.StringAttribute("layerPath", layerPath),
	)

	bytes, err := json.Marshal(layerData)
	if err != nil {
		return err
	}

	// Options are not used in the platform as of RS5
	err = hcsInitializeWritableLayer(layerPath, string(bytes), "")
	if err != nil {
		return errors.Wrap(err, "failed to intitialize container layer")
	}
	return nil
}
