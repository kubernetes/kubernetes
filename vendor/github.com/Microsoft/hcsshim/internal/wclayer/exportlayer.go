//go:build windows

package wclayer

import (
	"context"
	"os"
	"strings"

	"github.com/Microsoft/go-winio"
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
)

// ExportLayer will create a folder at exportFolderPath and fill that folder with
// the transport format version of the layer identified by layerId. This transport
// format includes any metadata required for later importing the layer (using
// ImportLayer), and requires the full list of parent layer paths in order to
// perform the export.
func ExportLayer(ctx context.Context, path string, exportFolderPath string, parentLayerPaths []string) (err error) {
	title := "hcsshim::ExportLayer"
	ctx, span := oc.StartSpan(ctx, title)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(
		trace.StringAttribute("path", path),
		trace.StringAttribute("exportFolderPath", exportFolderPath),
		trace.StringAttribute("parentLayerPaths", strings.Join(parentLayerPaths, ", ")))

	// Generate layer descriptors
	layers, err := layerPathsToDescriptors(ctx, parentLayerPaths)
	if err != nil {
		return err
	}

	err = exportLayer(&stdDriverInfo, path, exportFolderPath, layers)
	if err != nil {
		return hcserror.New(err, title, "")
	}
	return nil
}

// LayerReader is an interface that supports reading an existing container image layer.
type LayerReader interface {
	// Next advances to the next file and returns the name, size, and file info
	Next() (string, int64, *winio.FileBasicInfo, error)
	// LinkInfo returns the number of links and the file identifier for the current file.
	LinkInfo() (uint32, *winio.FileIDInfo, error)
	// Read reads data from the current file, in the format of a Win32 backup stream, and
	// returns the number of bytes read.
	Read(b []byte) (int, error)
	// Close finishes the layer reading process and releases any resources.
	Close() error
}

// NewLayerReader returns a new layer reader for reading the contents of an on-disk layer.
// The caller must have taken the SeBackupPrivilege privilege
// to call this and any methods on the resulting LayerReader.
func NewLayerReader(ctx context.Context, path string, parentLayerPaths []string) (_ LayerReader, err error) {
	ctx, span := oc.StartSpan(ctx, "hcsshim::NewLayerReader")
	defer func() {
		if err != nil {
			oc.SetSpanStatus(span, err)
			span.End()
		}
	}()
	span.AddAttributes(
		trace.StringAttribute("path", path),
		trace.StringAttribute("parentLayerPaths", strings.Join(parentLayerPaths, ", ")))

	if len(parentLayerPaths) == 0 {
		// This is a base layer. It gets exported differently.
		return newBaseLayerReader(path, span), nil
	}

	exportPath, err := os.MkdirTemp("", "hcs")
	if err != nil {
		return nil, err
	}
	err = ExportLayer(ctx, path, exportPath, parentLayerPaths)
	if err != nil {
		os.RemoveAll(exportPath)
		return nil, err
	}
	return &legacyLayerReaderWrapper{
		ctx:               ctx,
		s:                 span,
		legacyLayerReader: newLegacyLayerReader(exportPath),
	}, nil
}

type legacyLayerReaderWrapper struct {
	ctx context.Context
	s   *trace.Span

	*legacyLayerReader
}

func (r *legacyLayerReaderWrapper) Close() (err error) {
	defer r.s.End()
	defer func() { oc.SetSpanStatus(r.s, err) }()

	err = r.legacyLayerReader.Close()
	os.RemoveAll(r.root)
	return err
}
