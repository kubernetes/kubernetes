//go:build windows

package computestorage

import (
	"context"

	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/pkg/errors"
	"golang.org/x/sys/windows"
)

// FormatWritableLayerVhd formats a virtual disk for use as a writable container layer.
//
// If the VHD is not mounted it will be temporarily mounted.
//
// NOTE: This API had a breaking change in the operating system after Windows Server 2019.
// On ws2019 the API expects to get passed a file handle from CreateFile for the vhd that
// the caller wants to format. On > ws2019, its expected that the caller passes a vhd handle
// that can be obtained from the virtdisk APIs.
func FormatWritableLayerVhd(ctx context.Context, vhdHandle windows.Handle) (err error) {
	title := "hcsshim::FormatWritableLayerVhd"
	ctx, span := oc.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()

	err = hcsFormatWritableLayerVhd(vhdHandle)
	if err != nil {
		return errors.Wrap(err, "failed to format writable layer vhd")
	}
	return nil
}
