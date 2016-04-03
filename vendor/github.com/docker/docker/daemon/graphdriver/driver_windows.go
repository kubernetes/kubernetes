package graphdriver

import (
	"github.com/docker/docker/pkg/archive"
	"github.com/microsoft/hcsshim"
)

type WindowsGraphDriver interface {
	Driver
	CopyDiff(id, sourceId string, parentLayerPaths []string) error
	LayerIdsToPaths(ids []string) []string
	Info() hcsshim.DriverInfo
	Export(id string, parentLayerPaths []string) (archive.Archive, error)
	Import(id string, layerData archive.ArchiveReader, parentLayerPaths []string) (int64, error)
}

var (
	// Slice of drivers that should be used in order
	priority = []string{
		"windowsfilter",
		"windowsdiff",
		"vfs",
	}
)

func GetFSMagic(rootpath string) (FsMagic, error) {
	// Note it is OK to return FsMagicUnsupported on Windows.
	return FsMagicUnsupported, nil
}
