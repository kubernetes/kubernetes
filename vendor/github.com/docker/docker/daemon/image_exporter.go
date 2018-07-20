package daemon

import (
	"io"
	"runtime"

	"github.com/docker/docker/image/tarexport"
	"github.com/docker/docker/pkg/system"
)

// ExportImage exports a list of images to the given output stream. The
// exported images are archived into a tar when written to the output
// stream. All images with the given tag and all versions containing
// the same tag are exported. names is the set of tags to export, and
// outStream is the writer which the images are written to.
func (daemon *Daemon) ExportImage(names []string, outStream io.Writer) error {
	// TODO @jhowardmsft LCOW. This will need revisiting later.
	platform := runtime.GOOS
	if system.LCOWSupported() {
		platform = "linux"
	}
	imageExporter := tarexport.NewTarExporter(daemon.stores[platform].imageStore, daemon.stores[platform].layerStore, daemon.referenceStore, daemon)
	return imageExporter.Save(names, outStream)
}

// LoadImage uploads a set of images into the repository. This is the
// complement of ImageExport.  The input stream is an uncompressed tar
// ball containing images and metadata.
func (daemon *Daemon) LoadImage(inTar io.ReadCloser, outStream io.Writer, quiet bool) error {
	// TODO @jhowardmsft LCOW. This will need revisiting later.
	platform := runtime.GOOS
	if system.LCOWSupported() {
		platform = "linux"
	}
	imageExporter := tarexport.NewTarExporter(daemon.stores[platform].imageStore, daemon.stores[platform].layerStore, daemon.referenceStore, daemon)
	return imageExporter.Load(inTar, outStream, quiet)
}
