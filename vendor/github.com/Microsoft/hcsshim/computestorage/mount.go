package computestorage

import (
	"context"
	"fmt"

	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
	"golang.org/x/sys/windows"
)

// GetLayerVhdMountPath returns the volume path for a virtual disk of a writable container layer.
func GetLayerVhdMountPath(ctx context.Context, vhdHandle windows.Handle) (path string, err error) {
	title := "hcsshim.GetLayerVhdMountPath"
	ctx, span := trace.StartSpan(ctx, title)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()

	var mountPath *uint16
	err = hcsGetLayerVhdMountPath(vhdHandle, &mountPath)
	if err != nil {
		return "", fmt.Errorf("failed to get vhd mount path: %s", err)
	}
	path = interop.ConvertAndFreeCoTaskMemString(mountPath)
	return path, nil
}
