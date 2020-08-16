package wclayer

import (
	"context"
	"os"

	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
)

// ProcessBaseLayer post-processes a base layer that has had its files extracted.
// The files should have been extracted to <path>\Files.
func ProcessBaseLayer(ctx context.Context, path string) (err error) {
	title := "hcsshim::ProcessBaseLayer"
	ctx, span := trace.StartSpan(ctx, title)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("path", path))

	err = processBaseImage(path)
	if err != nil {
		return &os.PathError{Op: title, Path: path, Err: err}
	}
	return nil
}

// ProcessUtilityVMImage post-processes a utility VM image that has had its files extracted.
// The files should have been extracted to <path>\Files.
func ProcessUtilityVMImage(ctx context.Context, path string) (err error) {
	title := "hcsshim::ProcessUtilityVMImage"
	ctx, span := trace.StartSpan(ctx, title)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("path", path))

	err = processUtilityImage(path)
	if err != nil {
		return &os.PathError{Op: title, Path: path, Err: err}
	}
	return nil
}
