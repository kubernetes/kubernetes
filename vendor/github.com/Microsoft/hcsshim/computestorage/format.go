package computestorage

import (
	"context"

	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/pkg/errors"
	"go.opencensus.io/trace"
	"golang.org/x/sys/windows"
)

// FormatWritableLayerVhd formats a virtual disk for use as a writable container layer.
//
// If the VHD is not mounted it will be temporarily mounted.
func FormatWritableLayerVhd(ctx context.Context, vhdHandle windows.Handle) (err error) {
	title := "hcsshim.FormatWritableLayerVhd"
	ctx, span := trace.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()

	err = hcsFormatWritableLayerVhd(vhdHandle)
	if err != nil {
		return errors.Wrap(err, "failed to format writable layer vhd")
	}
	return nil
}
