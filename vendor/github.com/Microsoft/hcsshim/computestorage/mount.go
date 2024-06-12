//go:build windows

package computestorage

import (
	"context"

	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/pkg/errors"
	"golang.org/x/sys/windows"
)

// GetLayerVhdMountPath returns the volume path for a virtual disk of a writable container layer.
func GetLayerVhdMountPath(ctx context.Context, vhdHandle windows.Handle) (path string, err error) {
	title := "hcsshim::GetLayerVhdMountPath"
	ctx, span := oc.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()

	var mountPath *uint16
	err = hcsGetLayerVhdMountPath(vhdHandle, &mountPath)
	if err != nil {
		return "", errors.Wrap(err, "failed to get vhd mount path")
	}
	path = interop.ConvertAndFreeCoTaskMemString(mountPath)
	return path, nil
}
