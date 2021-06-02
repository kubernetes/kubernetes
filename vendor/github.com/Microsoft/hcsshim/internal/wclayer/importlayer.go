package wclayer

import (
	"context"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/Microsoft/go-winio"
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/Microsoft/hcsshim/internal/safefile"
	"go.opencensus.io/trace"
)

// ImportLayer will take the contents of the folder at importFolderPath and import
// that into a layer with the id layerId.  Note that in order to correctly populate
// the layer and interperet the transport format, all parent layers must already
// be present on the system at the paths provided in parentLayerPaths.
func ImportLayer(ctx context.Context, path string, importFolderPath string, parentLayerPaths []string) (err error) {
	title := "hcsshim::ImportLayer"
	ctx, span := trace.StartSpan(ctx, title)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(
		trace.StringAttribute("path", path),
		trace.StringAttribute("importFolderPath", importFolderPath),
		trace.StringAttribute("parentLayerPaths", strings.Join(parentLayerPaths, ", ")))

	// Generate layer descriptors
	layers, err := layerPathsToDescriptors(ctx, parentLayerPaths)
	if err != nil {
		return err
	}

	err = importLayer(&stdDriverInfo, path, importFolderPath, layers)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}

// LayerWriter is an interface that supports writing a new container image layer.
type LayerWriter interface {
	// Add adds a file to the layer with given metadata.
	Add(name string, fileInfo *winio.FileBasicInfo) error
	// AddLink adds a hard link to the layer. The target must already have been added.
	AddLink(name string, target string) error
	// Remove removes a file that was present in a parent layer from the layer.
	Remove(name string) error
	// Write writes data to the current file. The data must be in the format of a Win32
	// backup stream.
	Write(b []byte) (int, error)
	// Close finishes the layer writing process and releases any resources.
	Close() error
}

type legacyLayerWriterWrapper struct {
	ctx context.Context
	s   *trace.Span

	*legacyLayerWriter
	path             string
	parentLayerPaths []string
}

func (r *legacyLayerWriterWrapper) Close() (err error) {
	defer r.s.End()
	defer func() { oc.SetSpanStatus(r.s, err) }()
	defer os.RemoveAll(r.root.Name())
	defer r.legacyLayerWriter.CloseRoots()

	err = r.legacyLayerWriter.Close()
	if err != nil {
		return err
	}

	if err = ImportLayer(r.ctx, r.destRoot.Name(), r.path, r.parentLayerPaths); err != nil {
		return err
	}
	for _, name := range r.Tombstones {
		if err = safefile.RemoveRelative(name, r.destRoot); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	// Add any hard links that were collected.
	for _, lnk := range r.PendingLinks {
		if err = safefile.RemoveRelative(lnk.Path, r.destRoot); err != nil && !os.IsNotExist(err) {
			return err
		}
		if err = safefile.LinkRelative(lnk.Target, lnk.TargetRoot, lnk.Path, r.destRoot); err != nil {
			return err
		}
	}

	// The reapplyDirectoryTimes must be called AFTER we are done with Tombstone
	// deletion and hard link creation. This is because Tombstone deletion and hard link
	// creation updates the directory last write timestamps so that will change the
	// timestamps added by the `Add` call. Some container applications depend on the
	// correctness of these timestamps and so we should change the timestamps back to
	// the original value (i.e the value provided in the Add call) after this
	// processing is done.
	err = reapplyDirectoryTimes(r.destRoot, r.changedDi)
	if err != nil {
		return err
	}

	// Prepare the utility VM for use if one is present in the layer.
	if r.HasUtilityVM {
		err := safefile.EnsureNotReparsePointRelative("UtilityVM", r.destRoot)
		if err != nil {
			return err
		}
		err = ProcessUtilityVMImage(r.ctx, filepath.Join(r.destRoot.Name(), "UtilityVM"))
		if err != nil {
			return err
		}
	}
	return nil
}

// NewLayerWriter returns a new layer writer for creating a layer on disk.
// The caller must have taken the SeBackupPrivilege and SeRestorePrivilege privileges
// to call this and any methods on the resulting LayerWriter.
func NewLayerWriter(ctx context.Context, path string, parentLayerPaths []string) (_ LayerWriter, err error) {
	ctx, span := trace.StartSpan(ctx, "hcsshim::NewLayerWriter")
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
		// This is a base layer. It gets imported differently.
		f, err := safefile.OpenRoot(path)
		if err != nil {
			return nil, err
		}
		return &baseLayerWriter{
			ctx:  ctx,
			s:    span,
			root: f,
		}, nil
	}

	importPath, err := ioutil.TempDir("", "hcs")
	if err != nil {
		return nil, err
	}
	w, err := newLegacyLayerWriter(importPath, parentLayerPaths, path)
	if err != nil {
		return nil, err
	}
	return &legacyLayerWriterWrapper{
		ctx:               ctx,
		s:                 span,
		legacyLayerWriter: w,
		path:              importPath,
		parentLayerPaths:  parentLayerPaths,
	}, nil
}
